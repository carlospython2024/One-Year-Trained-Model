from __future__ import annotations

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
import requests
import os
import hmac
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler, label_binarize
import os, json, uuid, time
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Literal, Tuple

from math import sqrt

import warnings
import smtplib
from email.mime.text import MIMEText
import pytz

load_dotenv()

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
api_key_roge = os.getenv("API_KEY_ROGE")
api_secret_roge = os.getenv("API_SECRET_ROGE")

email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")

EMAIL_SMTP = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_RECEIVER = "candrelosolaz@gmail.com"


RESTART_HOUR_1 = 8
RESTART_HOUR_2 = 12
RESTART_HOUR_3 = 17
RESTART_HOUR_4 = 22

warnings.filterwarnings('ignore')

@dataclass
class Fees:
    """Estructura de comisiones. Todas en proporción (p.ej., 0.00055 = 0.055%)."""
    taker_open: float = 0.00055
    taker_close: float = 0.00055
    maker_open: float = 0.0002
    maker_close: float = 0.0002


@dataclass
class Trade:
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_price: float
    tp_price: float
    sl_price: float
    qty: float  # contratos (para lineales USDT: qty = notional / price)
    leverage: float
    opened_with: Literal["taker", "maker"] = "taker"
    closed_with: Literal["taker", "maker"] = "taker"

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[Literal["tp", "sl", "close"]] = None

    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl_usdt(self) -> float:
        if self.exit_price is None:
            return 0.0
        # Lineal USDT: PnL = (Px_exit - Px_entry) * qty (para long), invertido para short
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.exit_price) * self.qty

    def fees_usdt(self) -> float:
        # Comisiones sobre el nocional (precio * qty)
        notional_entry = self.entry_price * self.qty
        notional_exit = (self.exit_price if self.exit_price is not None else self.entry_price) * self.qty
        open_fee = notional_entry * (self.opened_with == "taker") * fees.taker_open + \
                   notional_entry * (self.opened_with == "maker") * fees.maker_open
        close_fee = notional_exit * (self.closed_with == "taker") * fees.taker_close + \
                    notional_exit * (self.closed_with == "maker") * fees.maker_close
        return float(open_fee + close_fee)


# Variable global simple para que Trade.fees_usdt tenga acceso a la instancia Fees
fees = Fees()


class TradeSimulator:
    """
    Simulador walk-forward para una clase externa `StochasticAnalyzer` del usuario.

    Flujo (v1.2):
      1) Entrena **una sola vez** con los datos **previos al inicio del período de test** (por defecto).
         *Opcional:* si `retrain_each_bar=True`, reentrena en cada vela.
      2) Para cada vela del rango de test, obtiene la señal. Por defecto usa un **predictor offline**
         (sin tocar el API) que toma una ventana de `offline_lookback_bars` y replica tu `predict_realtime`
         usando únicamente el dataset local.
      3) Si aprueba validaciones estilo `monitor_predictions`, abre trade.
      4) Cierra por TP/SL usando high/low y aplica fees. Puedes forzar que el cierre no pase de `test_end`.
    """

    def __init__(
        self,
        analyzer,  # instancia de StochasticAnalyzer (ya creada por el usuario)
        candles: pd.DataFrame,
        symbol: str = "AVAXUSDT",
        interval: str = "5",
        initial_equity: float = 1000.0,
        leverage: float = 10.0,
        risk_fraction: float = 0.01,
        bybit_fees: Fees = Fees(),
        proba_threshold_day: float = 0.56,
        proba_threshold_night: float = 0.60,
        day_hours: Tuple[dtime, dtime] = (dtime(8, 0), dtime(20, 0)),
        order_conflict_policy: Literal["worst", "best", "tp_first", "sl_first"] = "worst",
        position_mode: Literal["fixed_risk", "fixed_notional"] = "fixed_risk",
        fixed_notional_usdt: float = 100.0,
        use_adapter_tp_sl_when_missing: bool = True,
        atr_period: int = 14,
        atr_mult_tp: float = 1.0,
        atr_mult_sl: float = 1.0,
        retrain_each_bar: bool = False,
        suppress_fit_stdout: bool = True,
        suppress_predict_stdout: bool = True,
        use_offline_predictor: bool = True,
        offline_lookback_bars: int = 200,
        thresholds: Optional[Dict[str, float]] = None,
        n_keep_for_fit: int = 40000,
        close_trades_at_test_end: bool = True,
    ):
        assert {"datetime", "open", "high", "low", "close"}.issubset(candles.columns), \
            "El DataFrame de velas debe contener: datetime, open, high, low, close"
        self.analyzer = analyzer
        self.df = candles.reset_index(drop=True).copy()
        if not np.issubdtype(self.df["datetime"].dtype, np.datetime64):
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], utc=False)
        self.symbol = symbol
        self.interval = interval
        self.equity = float(initial_equity)
        self.leverage = float(leverage)
        self.risk_fraction = float(risk_fraction)
        self.fees = bybit_fees
        self.proba_threshold_day = proba_threshold_day
        self.proba_threshold_night = proba_threshold_night
        self.day_hours = day_hours
        self.order_conflict_policy = order_conflict_policy
        self.position_mode = position_mode
        self.fixed_notional_usdt = fixed_notional_usdt
        self.use_adapter_tp_sl_when_missing = use_adapter_tp_sl_when_missing
        self.atr_period = int(atr_period)
        self.atr_mult_tp = float(atr_mult_tp)
        self.atr_mult_sl = float(atr_mult_sl)
        self.retrain_each_bar = bool(retrain_each_bar)
        self.suppress_fit_stdout = bool(suppress_fit_stdout)
        self.suppress_predict_stdout = bool(suppress_predict_stdout)
        self.use_offline_predictor = bool(use_offline_predictor)
        self.offline_lookback_bars = int(offline_lookback_bars)
        self.thresholds = thresholds or {"bearish": 0.55, "bullish": 0.55, "lateral": 0.40}
        self.n_keep_for_fit = int(n_keep_for_fit)
        self.close_trades_at_test_end = bool(close_trades_at_test_end)

        global fees
        fees = bybit_fees

        # ATR para fallback TP/SL
        self.df["tr"] = self._true_range(self.df)
        self.df["atr"] = self.df["tr"].rolling(self.atr_period).mean()

        self.trades: List[Trade] = []
        self.logs: List[str] = []

    # --------------------------
    # ADAPTERS / INTEGRACIONES
    # --------------------------

    def fit_model(self, train_df: pd.DataFrame):
        """Entrena el modelo del usuario **solo** con `train_df`.
        Seteamos `analyzer.data = train_df` y llamamos `create_predictive_model` con N_KEEP capado.
        """
        self.analyzer.data = train_df.copy()
        N_KEEP = min(len(train_df), self.n_keep_for_fit)
        if self.suppress_fit_stdout:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.analyzer.create_predictive_model(display_graphic=False, N_KEEP=N_KEEP)
        else:
            self.analyzer.create_predictive_model(display_graphic=False, N_KEEP=N_KEEP)

    def _extract_proba_for_monitor(self, sig: Dict) -> Optional[float]:
        """Toma la probabilidad relevante para evaluar umbrales de monitor.
        Acepta tanto 'proba' como 'probabilities'. Si hay 'prediction', usa la de esa clase.
        """
        if sig is None:
            return None
        if "proba" in sig and sig["proba"] is not None:
            try:
                return float(sig["proba"])
            except Exception:
                pass
        probs = sig.get("probabilities") or sig.get("probs")
        pred = sig.get("prediction")
        if probs and isinstance(probs, dict):
            if pred in probs:
                return float(probs[pred])
            # si no hay pred, usa la mayor
            try:
                return float(max(probs.values()))
            except Exception:
                return None
        return None

    def offline_predict_at_index(self, i: int) -> Optional[Dict]:
        """Replica `predict_realtime` pero **offline** usando el dataset local.
        Usa una ventana de `offline_lookback_bars` terminando en el índice `i` (incluido).
        """
        start_i = max(0, i - self.offline_lookback_bars + 1)
        df_win = self.df.loc[start_i:i, ["datetime", "open", "high", "low", "close", "volume"]].copy()
        if len(df_win) < 100:
            return None

        # indicadores & features con las funciones del analyzer
        if self.suppress_predict_stdout:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                df_win = self.analyzer.add_indicators(df_win)
                df_win = self.analyzer.add_volatility_zones(df_win)
                df_ready, features = self.analyzer.preparar_features(df_win)
        else:
            df_win = self.analyzer.add_indicators(df_win)
            df_win = self.analyzer.add_volatility_zones(df_win)
            df_ready, features = self.analyzer.preparar_features(df_win)

        if df_ready.empty:
            return None
        last_row_feat = df_ready[features].iloc[-1:]
        last_meta = df_win.iloc[-1]

        # escalar + predecir
        last_scaled = self.analyzer.scaler.transform(last_row_feat)
        proba_vec = self.analyzer.model.predict_proba(last_scaled)[0]
        classes = list(self.analyzer.model.classes_)
        probabilities = {cls: float(round(p, 4)) for cls, p in zip(classes, proba_vec)}

        # aplicar thresholds del usuario (si tiene helper interno)
        if hasattr(self.analyzer, "_apply_thresholds"):
            pred_class = self.analyzer._apply_thresholds(
                proba_vec, classes, self.thresholds,
                default_class="lateral", global_floor=0.51, min_margin=0.03
            )
        else:
            # fallback simple
            best_idx = int(np.argmax(proba_vec))
            pred_class = classes[best_idx]

        # filtro de “fake” señales como en tu predict_realtime
        ema_10 = float(last_meta.get("EMA_10", np.nan))
        close_px = float(last_meta["close"])
        umbral_strong = getattr(self.analyzer, "umbral_strong", 0.84)
        pred = pred_class
        if pred_class == 'bullish' and (not np.isnan(ema_10)) and close_px < ema_10 and probabilities.get('bullish', 0) < umbral_strong:
            pred = 'lateral'
        elif pred_class == 'bearish' and (not np.isnan(ema_10)) and close_px > ema_10 and probabilities.get('bearish', 0) < umbral_strong:
            pred = 'lateral'

        return {
            "prediction": pred,
            "probabilities": probabilities,
            "timestamp": last_meta['datetime'],
            "last_row_close": close_px,
            "last_row_atr": float(last_meta.get('ATR', np.nan)),
        }

    def get_signal(self, asof_index: int) -> Optional[Dict]:
        """Obtiene la predicción para la vela `asof_index`.
        Por defecto usa el predictor offline. Si `use_offline_predictor=False`, llama a `predict_realtime`.
        """
        if self.use_offline_predictor:
            return self.offline_predict_at_index(asof_index)
        else:
            # Online: el método del usuario podría imprimir; respetamos suppress_predict_stdout
            if self.suppress_predict_stdout:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    return self.analyzer.predict_realtime(symbol=self.symbol, interval=self.interval)
            return self.analyzer.predict_realtime(symbol=self.symbol, interval=self.interval)

    # --------------------------
    # LÓGICA DE NEGOCIO
    # --------------------------

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        hl = df["high"] - df["low"]
        hc = (df["high"] - prev_close).abs()
        lc = (df["low"] - prev_close).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    def _passes_monitor(self, ts: pd.Timestamp, proba: float) -> bool:
        start, end = self.day_hours
        in_day = start <= ts.time() <= end
        thresh = self.proba_threshold_day if in_day else self.proba_threshold_night
        return proba is not None and float(proba) >= thresh

    def _tp_sl_from_adapter_or_atr(self, i: int, side: Literal["long", "short"],
                                   maybe_tp: Optional[float], maybe_sl: Optional[float]) -> Tuple[float, float]:
        px = float(self.df.loc[i, "close"])
        if self.use_adapter_tp_sl_when_missing and (maybe_tp is not None) and (maybe_sl is not None):
            return float(maybe_tp), float(maybe_sl)
        atr = float(self.df.loc[i, "atr"]) if not np.isnan(self.df.loc[i, "atr"]) else px * 0.005
        if side == "long":
            tp = px + self.atr_mult_tp * atr
            sl = px - self.atr_mult_sl * atr
        else:
            tp = px - self.atr_mult_tp * atr
            sl = px + self.atr_mult_sl * atr
        return tp, sl

    def _size_position(self, px: float, kelly_fraction: Optional[float] = None) -> float:
        k = max(0.0, min(1.0, float(kelly_fraction))) if kelly_fraction is not None else 1.0
        if self.position_mode == "fixed_risk":
            margin = self.equity * self.risk_fraction * k
            notional = margin * self.leverage
        else:
            notional = self.fixed_notional_usdt * k
        qty = max(0.0, notional / px)
        return qty

    def _apply_conflict_policy(self, bar_high: float, bar_low: float, tp: float, sl: float, side: str) -> Optional[str]:
        hit_tp = bar_high >= tp if side == "long" else bar_low <= tp
        hit_sl = bar_low <= sl if side == "long" else bar_high >= sl
        if hit_tp and hit_sl:
            if self.order_conflict_policy == "tp_first":
                return "tp"
            elif self.order_conflict_policy == "sl_first":
                return "sl"
            elif self.order_conflict_policy == "best":
                return "tp"
            else:
                return "sl"
        elif hit_tp:
            return "tp"
        elif hit_sl:
            return "sl"
        return None

    def _try_close_trade_on_bar(self, trade: Trade, i: int) -> None:
        if not trade.is_open():
            return
        bar = self.df.loc[i]
        decision = self._apply_conflict_policy(bar.high, bar.low, trade.tp_price, trade.sl_price, trade.side)
        if decision is not None:
            trade.exit_time = bar["datetime"]
            trade.exit_reason = decision
            trade.exit_price = trade.tp_price if decision == "tp" else trade.sl_price
            trade.closed_with = "taker"

    # --------------------------
    # LOOP PRINCIPAL
    # --------------------------

    def run(self, test_start: str, test_end: Optional[str] = None) -> Dict:
        """
        Ejecuta el walk-forward entre [test_start, test_end].
        Por defecto **entrena una sola vez** con el histórico previo a `test_start`.
        Si `retrain_each_bar=True`, reentrena en cada barra antes de predecir.
        """
        mask = (self.df["datetime"] >= pd.to_datetime(test_start))
        if test_end is not None:
            mask &= (self.df["datetime"] <= pd.to_datetime(test_end))
        idxs = self.df.index[mask].tolist()
        if not idxs:
            raise ValueError("Rango de test vacío. Revisa las fechas.")

        # ENTRENAR UNA VEZ con datos previos a test_start
        first_test_i = idxs[0]
        train_slice_initial = self.df.loc[: first_test_i - 1]
        if len(train_slice_initial) >= 200:
            self.fit_model(train_slice_initial)
        else:
            self.logs.append("Aviso: histórico previo insuficiente para entrenar antes del test.")

        current_trade: Optional[Trade] = None
        last_allowed_index = idxs[-1]

        for pos, i in enumerate(idxs):
            # (Opcional) reentrenar por barra
            if self.retrain_each_bar:
                train_slice = self.df.loc[: i - 1]
                if len(train_slice) < 200:
                    continue
                self.fit_model(train_slice)

            # Si hay trade abierto, intentar cierre en esta barra
            if current_trade is not None and current_trade.is_open():
                self._try_close_trade_on_bar(current_trade, i)
                if not current_trade.is_open():
                    pnl = current_trade.pnl_usdt()
                    fees_paid = current_trade.fees_usdt()
                    self.equity += pnl - fees_paid
                    self.trades.append(current_trade)
                    current_trade = None

            # Si no hay trade, evaluar nueva señal
            if current_trade is None:
                sig = self.get_signal(i)
                if not sig:
                    continue
                pred = sig.get("prediction")
                proba = self._extract_proba_for_monitor(sig)
                if pred not in {"bullish", "bearish"}:
                    continue
                ts = self.df.loc[i, "datetime"]
                if not self._passes_monitor(ts, proba):
                    continue

                px = float(self.df.loc[i, "close"])  # entrada al cierre de i
                kelly = sig.get("kelly_fraction")
                qty = self._size_position(px, kelly)
                if qty <= 0:
                    continue
                side = "long" if pred == "bullish" else "short"
                tp, sl = self._tp_sl_from_adapter_or_atr(i, side, sig.get("take_profit") or sig.get("tp"),
                                                          sig.get("stop_loss") or sig.get("sl"))
                current_trade = Trade(
                    side=side,
                    entry_time=ts,
                    entry_price=px,
                    tp_price=float(tp),
                    sl_price=float(sl),
                    qty=float(qty),
                    leverage=self.leverage,
                    opened_with="taker",
                    closed_with="taker",
                )
                # comisión de entrada
                entry_fee = px * qty * (self.fees.taker_open if current_trade.opened_with == "taker" else self.fees.maker_open)
                self.equity -= entry_fee

            # Avanzar para cierre, pero no más allá de `test_end` si se pidió
            j_limit = last_allowed_index if self.close_trades_at_test_end else len(self.df) - 1
            j = i + 1
            while current_trade is not None and current_trade.is_open() and j <= j_limit:
                self._try_close_trade_on_bar(current_trade, j)
                if not current_trade.is_open():
                    pnl = current_trade.pnl_usdt()
                    fees_paid = current_trade.fees_usdt()
                    self.equity += pnl - fees_paid
                    self.trades.append(current_trade)
                    current_trade = None
                    break
                j += 1

        # Si queda abierta al final, cerramos a mercado en la última barra del test
        if current_trade is not None and current_trade.is_open():
            last_bar = self.df.loc[last_allowed_index]
            current_trade.exit_time = last_bar["datetime"]
            current_trade.exit_price = float(last_bar["close"])
            current_trade.exit_reason = "close"
            pnl = current_trade.pnl_usdt()
            fees_paid = current_trade.fees_usdt()
            self.equity += pnl - fees_paid
            self.trades.append(current_trade)

        wins = sum(1 for t in self.trades if t.exit_reason == "tp")
        losses = sum(1 for t in self.trades if t.exit_reason == "sl")
        rr = wins / max(1, losses) if (wins + losses) > 0 else None

        return {
            "final_equity": float(self.equity),
            "n_trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_loss_ratio": float(rr) if rr is not None else None,
            "trades": [asdict(t) for t in self.trades],
            "fees": asdict(self.fees),
        }


def _now_iso():
    # guardamos en UTC con marca ISO + guardamos hora local aparte
    return datetime.now(timezone.utc).isoformat()

class TradeLogger:
    """
    Logger de trading estilo 'journal':
    - Cada evento se escribe en JSONL (append-only) para auditoría.
    - Mantiene un estado de posiciones abiertas en memoria para control rápido.
    - Permite calcular estadísticas a partir del log.
    """
    def __init__(self, log_path="trade_log.jsonl"):
        self.log_path = log_path
        # open_positions: key=(symbol), value=dict con datos de la posición abierta
        self.open_positions = {}  # { "AVAXUSDT": {...} }
        # para series de equity (si registras PnL intradía)
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                pass

    def _write(self, record: dict):
        record["ts_utc"] = _now_iso()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -------- Señales y órdenes --------
    def log_signal(self, *, symbol, timeframe, side, prob, margin_pp, probs_dict, kelly, rr, take_profit, stop_loss, price, reason, gate):
        """
        Registra la señal que gatilla la operación (antes de enviar orden).
        Devuelve un trade_id para seguir la trazabilidad.
        """
        trade_id = str(uuid.uuid4())
        rec = {
            "event": "signal",
            "trade_id": trade_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,                 # 'long'/'short'
            "prob": prob,
            "margin_pp": margin_pp,       # top1-top2 en puntos porcentuales
            "probs": probs_dict,
            "kelly": kelly,
            "reward_risk": rr,
            "entry_price": price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "gate": gate,                 # 'soft' / 'strong'
            "reason": reason              # texto libre: p.ej. "u_soft >= 0.80 & margin_ok"
        }
        self._write(rec)
        return trade_id

    def log_order_attempt(self, *, trade_id, symbol, side, qty, payload):
        self._write({
            "event": "order_attempt",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "payload": payload
        })

    def log_order_ack(self, *, trade_id, symbol, side, qty, broker_raw, order_id):
        """
        Acknowledge de orden aceptada por el broker.
        Marca posición como abierta.
        """
        self._write({
            "event": "order_ack",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_id": order_id,
            "broker_raw": broker_raw
        })
        # Guardamos estado de posición abierta (si no existía)
        # NOTA: asumimos una sola posición por símbolo (one-way). Si usas hedge-mode, habría que separar por side.
        self.open_positions[symbol] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_id": order_id,
            "opened_utc": _now_iso()
        }

    def log_order_reject(self, *, trade_id, symbol, side, qty, broker_raw):
        """
        Incluye casos como "Cambio de posición no permitido".
        """
        self._write({
            "event": "order_reject",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "broker_raw": broker_raw
        })

    # -------- Cierres y PnL --------
    def log_close(self, *, symbol, exit_reason, exit_price, pnl_usdt, pnl_r, order_id=None, trade_id=None):
        """
        Registra el cierre de la posición. Puedes llamarlo cuando detectes:
        - Size vuelve a 0 (poll al exchange)
        - Te llega webhook de TP/SL lleno
        - Cierre manual
        """
        # Busca trade_id si no lo pasaron
        if trade_id is None:
            info = self.open_positions.get(symbol)
            trade_id = info["trade_id"] if info else str(uuid.uuid4())

        rec = {
            "event": "close",
            "trade_id": trade_id,
            "symbol": symbol,
            "exit_reason": exit_reason,  # 'tp'/'sl'/'manual'/'error'
            "exit_price": exit_price,
            "pnl_usdt": pnl_usdt,
            "pnl_r": pnl_r,              # PnL en múltiplos de riesgo (R)
            "order_id": order_id
        }
        self._write(rec)
        # Limpia estado en memoria
        if symbol in self.open_positions:
            self.open_positions.pop(symbol, None)

    def log_info(self, **kwargs):
        rec = {"event": "info"}
        rec.update(kwargs)
        self._write(rec)

    # -------- Utilidades --------
    def has_open_position(self, symbol, side=None):
        """
        Retorna True si hay una posición abierta en 'symbol'.
        Si pasas 'side', valida que coincida.
        """
        info = self.open_positions.get(symbol)
        if not info:
            return False
        if side is None:
            return True
        return info.get("side") == side

    # -------- Estadísticas --------
    def compute_stats(self, pretty: bool = False, return_frames: bool = False):
        """
        Lee el JSONL y arma KPIs + breakdowns.
        - pretty=True  → retorna (stats_dict, pretty_text)
        - return_frames=True → además retorna (df_trades, df_side, df_gate)
        - por defecto retorna solo stats_dict

        Requisitos: pandas, numpy.
        """
        def _to_dt(x):
            try:
                return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
            except Exception:
                return pd.NaT

        def _pick(df, base):
            # Devuelve la columna si existe o su variante con sufijo _x/_y tras merge
            for c in (base, base + "_x", base + "_y"):
                if c in df.columns:
                    return c
            return None

        def _streaks(wins_series):
            # wins_series: 1=win, 0=loss
            best = worst = cur_w = cur_l = 0
            for w in wins_series.astype(int).tolist():
                if w == 1:
                    cur_w += 1;
                    best = max(best, cur_w)
                    cur_l = 0
                else:
                    cur_l += 1;
                    worst = max(worst, cur_l)
                    cur_w = 0
            return best, worst

        if not os.path.exists(self.log_path):
            return {"trades": 0}

        # --- Carga JSONL ---
        rows = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        if not rows:
            return {"trades": 0}

        df = pd.DataFrame(rows)

        # --- Filtra eventos relevantes ---
        closes = df[df["event"] == "close"].copy()
        signals = df[df["event"] == "signal"].copy()

        if closes.empty or signals.empty:
            out = {"trades": 0}
            return (out, "No hay trades cerrados.") if pretty else out

        # Dejar la PRIMERA señal por trade_id (origen del trade)
        signals = signals.sort_values("ts_utc").drop_duplicates(subset=["trade_id"], keep="first")

        # Selección robusta de columnas de signal
        sig_cols = ["trade_id", "symbol", "side", "reward_risk", "entry_price",
                    "take_profit", "stop_loss", "prob", "kelly", "gate", "ts_utc"]
        for c in sig_cols:
            if c not in signals.columns:
                signals[c] = np.nan
        signals = signals[sig_cols].rename(columns={"ts_utc": "signal_ts_utc"})

        # Merge close <- signal
        trades = closes.merge(signals, on="trade_id", how="left", suffixes=("", "_sig"))

        # Parseo de timestamps
        trades["close_ts_utc"] = trades["ts_utc"].apply(_to_dt)
        trades["signal_ts_utc"] = trades["signal_ts_utc"].apply(_to_dt)
        trades = trades.sort_values("close_ts_utc")

        # Duración (min)
        trades["duration_min"] = (trades["close_ts_utc"] - trades["signal_ts_utc"]).dt.total_seconds() / 60.0

        # Win/loss
        trades["win"] = (trades["pnl_usdt"] > 0).astype(int)
        sum_wins = trades.loc[trades["win"] == 1, "pnl_usdt"].sum()
        sum_losses = trades.loc[trades["win"] == 0, "pnl_usdt"].abs().sum()
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (np.inf if sum_wins > 0 else 0.0)
        winrate = trades["win"].mean() if len(trades) else 0.0

        # Expectancy en R
        has_r = "pnl_r" in trades.columns and trades["pnl_r"].notna().any()
        avg_r = trades["pnl_r"].mean() if has_r else np.nan
        expectancy_r = avg_r

        # Equity y DD
        trades["cum_pnl"] = trades["pnl_usdt"].cumsum()
        running_max = trades["cum_pnl"].cummax()
        drawdown = trades["cum_pnl"] - running_max
        max_dd = drawdown.min() if not drawdown.empty else 0.0

        # Métricas extra
        avg_win = trades.loc[trades["win"] == 1, "pnl_usdt"].mean()
        avg_loss = trades.loc[trades["win"] == 0, "pnl_usdt"].mean()
        med_pnl = trades["pnl_usdt"].median()
        best_trade = trades.loc[trades["pnl_usdt"].idxmax()] if not trades.empty else None
        worst_trade = trades.loc[trades["pnl_usdt"].idxmin()] if not trades.empty else None
        mean_dur = trades["duration_min"].mean()
        med_dur = trades["duration_min"].median()

        # Rachas
        best_streak, worst_streak = _streaks(trades["win"])

        # Sharpe por trade (no anualizado)
        mu = trades["pnl_usdt"].mean()
        sd = trades["pnl_usdt"].std(ddof=1)
        sharpe_tr = (mu / sd) if sd and not np.isnan(sd) and sd != 0 else np.nan

        total_pnl = trades["pnl_usdt"].sum()

        # Breakdown por side
        if "side" in trades.columns:
            side_grp = trades.groupby("side", dropna=False)["pnl_usdt"].agg(
                trades="count", pnl_total="sum", pnl_avg="mean"
            ).reset_index()
            # winrate por side
            wr_side = trades.groupby("side", dropna=False)["win"].mean().reset_index().rename(
                columns={"win": "winrate"})
            side_grp = side_grp.merge(wr_side, on="side", how="left")
        else:
            side_grp = pd.DataFrame()

        # Breakdown por gate
        if "gate" in trades.columns:
            gate_grp = trades.groupby("gate", dropna=False)["pnl_usdt"].agg(
                trades="count", pnl_total="sum", pnl_avg="mean"
            ).reset_index()
            wr_gate = trades.groupby("gate", dropna=False)["win"].mean().reset_index().rename(
                columns={"win": "winrate"})
            gate_grp = gate_grp.merge(wr_gate, on="gate", how="left")
        else:
            gate_grp = pd.DataFrame()

        # Salida principal (dict)
        out = {
            "trades": int(len(trades)),
            "winrate": float(winrate),
            "profit_factor": float(profit_factor),
            "avg_R": (float(avg_r) if not np.isnan(avg_r) else None),
            "expectancy_R": (float(expectancy_r) if not np.isnan(expectancy_r) else None),
            "avg_win_usdt": (float(avg_win) if not np.isnan(avg_win) else None),
            "avg_loss_usdt": (float(avg_loss) if not np.isnan(avg_loss) else None),
            "median_pnl_usdt": (float(med_pnl) if not np.isnan(med_pnl) else None),
            "max_drawdown_usdt": float(max_dd),
            "total_pnl_usdt": float(total_pnl),
            "mean_duration_min": (float(mean_dur) if not np.isnan(mean_dur) else None),
            "median_duration_min": (float(med_dur) if not np.isnan(med_dur) else None),
            "best_streak_wins": int(best_streak),
            "worst_streak_losses": int(worst_streak),
            "sharpe_per_trade": (float(sharpe_tr) if not np.isnan(sharpe_tr) else None),
            "best_trade": {
                "trade_id": best_trade["trade_id"],
                "pnl_usdt": float(best_trade["pnl_usdt"]),
                "close_ts_utc": str(best_trade["close_ts_utc"])
            } if best_trade is not None else None,
            "worst_trade": {
                "trade_id": worst_trade["trade_id"],
                "pnl_usdt": float(worst_trade["pnl_usdt"]),
                "close_ts_utc": str(worst_trade["close_ts_utc"])
            } if worst_trade is not None else None,
        }

        # Tabla bonita opcional (texto)
        if pretty:
            def f(v):
                if v is None: return "—"
                if isinstance(v, float):
                    return "∞" if np.isinf(v) else f"{v:.2f}"
                return str(v)

            lines = []
            lines.append("═ KPIs Generales ═")
            lines.append(f"Trades cerrados         : {out['trades']}")
            lines.append(f"Win rate                : {out['winrate'] * 100:.2f}%")
            lines.append(
                f"Profit factor           : {('∞' if np.isinf(out['profit_factor']) else f(out['profit_factor']))}")
            lines.append(f"Expectancy (R/trade)    : {f(out['expectancy_R'])}")
            lines.append(f"Avg R                   : {f(out['avg_R'])}")
            lines.append(f"Avg win (USDT)          : {f(out['avg_win_usdt'])}")
            lines.append(f"Avg loss (USDT)         : {f(out['avg_loss_usdt'])}")
            lines.append(f"Median PnL (USDT)       : {f(out['median_pnl_usdt'])}")
            lines.append(f"Max drawdown (USDT)     : {f(out['max_drawdown_usdt'])}")
            lines.append(f"PnL neto total (USDT)   : {f(out['total_pnl_usdt'])}")
            lines.append(f"Duración media (min)    : {f(out['mean_duration_min'])}")
            lines.append(f"Duración mediana (min)  : {f(out['median_duration_min'])}")
            lines.append(f"Mejor racha (wins)      : {out['best_streak_wins']}")
            lines.append(f"Peor racha (losses)     : {out['worst_streak_losses']}")
            lines.append(f"Sharpe por trade        : {f(out['sharpe_per_trade'])}")

            # Best/Worst
            if out["best_trade"]:
                lines.append(
                    f"Best trade              : {out['best_trade']['trade_id']}  ({out['best_trade']['pnl_usdt']:.2f} USDT)")
            if out["worst_trade"]:
                lines.append(
                    f"Worst trade             : {out['worst_trade']['trade_id']} ({out['worst_trade']['pnl_usdt']:.2f} USDT)")

            # Breakdown por side
            if not side_grp.empty:
                lines.append("\n─ Breakdown por side ─")
                lines.append("side   trades   winrate     pnl_total    pnl_avg")
                for _, r in side_grp.iterrows():
                    wr = f"{(r['winrate'] * 100):.1f}%"
                    lines.append(
                        f"{str(r['side']):5}  {int(r['trades']):6d}   {wr:>7}   {r['pnl_total']:.2f}    {r['pnl_avg']:.2f}")

            # Breakdown por gate
            if not gate_grp.empty:
                lines.append("\n─ Breakdown por gate ─")
                lines.append("gate   trades   winrate     pnl_total    pnl_avg")
                for _, r in gate_grp.iterrows():
                    g = str(r['gate']) if pd.notna(r['gate']) else "None"
                    wr = f"{(r['winrate'] * 100):.1f}%"
                    lines.append(f"{g:5}  {int(r['trades']):6d}   {wr:>7}   {r['pnl_total']:.2f}    {r['pnl_avg']:.2f}")

            pretty_text = "\n".join(lines)

        # Retornos según flags
        if pretty and return_frames:
            return out, pretty_text, trades, side_grp, gate_grp
        elif pretty:
            return out, pretty_text
        elif return_frames:
            return out, trades, side_grp, gate_grp
        else:
            return out

    def enrich_open_position(self, symbol, **kwargs):
        """Añade datos como entry_price, tp/sl, fee_rate, etc. a la posición abierta."""
        if symbol in self.open_positions:
            self.open_positions[symbol].update(kwargs)

    def get_open_position(self, symbol):
        return self.open_positions.get(symbol)

    def set_open_position(self, symbol, posdict):
        self.open_positions[symbol] = posdict


class StochasticAnalyzer:
    def __init__(self, api_key=None, api_secret=None):
        self.min_margin = 0.05
        self.data = None
        self.results = []
        self.model = None
        self.scaler = StandardScaler()
        self.api_key = api_key
        self.api_secret = api_secret
        self.atr_mult = 4
        self.atr_stop = 2.7
        self.lookahead = 70
        self.tradelogger = TradeLogger(log_path="trade_log.jsonl")

    def get_data_filename(self, symbol, interval, start_date, end_date):
        """
        Genera el nombre del archivo CSV basado en los parámetros
        """
        return f"{symbol}_{interval}m_{start_date}_{end_date}.csv"

    def load_cached_data(self, filename):
        """
        Carga datos desde archivo CSV si existe
        """
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['datetime'] = pd.to_datetime(df['datetime'])
                print(f"Datos cargados desde cache: {filename} ({len(df)} registros)")
                return df
            except Exception as e:
                print(f"Error cargando cache: {e}")
                return None
        return None

    def save_data_to_cache(self, df, filename):
        """
        Guarda datos en archivo CSV
        """
        try:
            df.to_csv(filename, index=False)
            print(f"Datos guardados en cache: {filename}")
        except Exception as e:
            print(f"Error guardando cache: {e}")

    def get_missing_date_ranges(self, cached_df, start_date, end_date):
        """
        Determina qué rangos de fechas faltan en los datos cacheados
        """
        if cached_df is None or len(cached_df) == 0:
            return [(start_date, end_date)]

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Fechas en cache
        cached_start = cached_df['datetime'].min()
        cached_end = cached_df['datetime'].max()

        missing_ranges = []

        # Verificar si necesitamos datos antes del cache
        if start_dt < cached_start:
            missing_ranges.append((start_date, cached_start.strftime("%Y-%m-%d")))

        # Verificar si necesitamos datos después del cache
        if end_dt > cached_end:
            missing_ranges.append((cached_end.strftime("%Y-%m-%d"), end_date))

        return missing_ranges

    def generate_signature(self, params, api_secret):
        """
        Genera la firma para autenticación con API privada de Bybit
        """
        param_str = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
        return hmac.new(api_secret.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()

    def get_bybit_data_private(
            self,
            symbol: str = "AVAXUSDT",
            interval: str = "5",
            start_date: str | None = "2025-06-01",
            end_date: str | None = "2025-06-02",
            limit: int = 0
    ):
        """
        Descarga velas OHLC desde la API PRIVADA de Bybit.
        ▸ limit == 0  → modo RANGO (usa start_date y end_date).
        ▸ limit >  0  → modo LOOK-BACK (últimas «limit» velas).
        """

        if not self.api_key or not self.api_secret:
            print("API Key y Secret requeridos para API privada → usando API pública")
            return self.get_bybit_data_public(symbol, interval, start_date, end_date, limit)

        url = "https://api.bybit.com/v5/market/kline"
        max_limit = 1000
        all_data = []

        if limit > 0:
            params = {
                "api_key": self.api_key,
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": str(min(limit, max_limit)),
                "timestamp": str(int(time.time() * 1000))
            }
            params["sign"] = self.generate_signature(params, self.api_secret)

            try:
                r = requests.get(url, params=params, timeout=10)
                data = r.json()
                if data["retCode"] == 0 and data["result"]["list"]:
                    for k in data["result"]["list"]:
                        all_data.append({
                            "timestamp": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        })
                else:
                    print(f"Error en respuesta: {data}")
            except Exception as e:
                print(f"Error (look-back): {e}")

        else:
            print(f"Obteniendo datos de {symbol} (API privada) desde {start_date} hasta {end_date}…")
            if not start_date or not end_date:
                raise ValueError("Debes especificar start_date y end_date en modo rango")

            # Calcular número de velas a descargar según el intervalo
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            minutes_per_candle = int(interval)
            num_candles = int((end_ts - start_ts) / (minutes_per_candle * 60 * 1000))

            # Si cabe en una sola petición
            if num_candles <= max_limit:
                params = {
                    "api_key": self.api_key,
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": str(start_ts),
                    "end": str(end_ts),
                    "limit": str(num_candles),
                    "timestamp": str(int(time.time() * 1000))
                }
                params["sign"] = self.generate_signature(params, self.api_secret)
                try:
                    r = requests.get(url, params=params, timeout=10)
                    data = r.json()
                    if data["retCode"] == 0 and data["result"]["list"]:
                        for k in data["result"]["list"]:
                            all_data.append({
                                "timestamp": int(k[0]),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5])
                            })
                    else:
                        print(f"Error en respuesta: {data}")
                except Exception as e:
                    print(f"Error (rango simple): {e}")
            else:
                # Usar bucle como antes
                current = start_ts
                while current < end_ts:
                    params = {
                        "api_key": self.api_key,
                        "category": "linear",
                        "symbol": symbol,
                        "interval": interval,
                        "start": str(current),
                        "limit": str(max_limit),
                        "timestamp": str(int(time.time() * 1000))
                    }
                    params["sign"] = self.generate_signature(params, self.api_secret)

                    try:
                        r = requests.get(url, params=params, timeout=10)
                        data = r.json()
                        if data["retCode"] == 0 and data["result"]["list"]:
                            klines = data["result"]["list"]
                            for k in klines:
                                all_data.append({
                                    "timestamp": int(k[0]),
                                    "open": float(k[1]),
                                    "high": float(k[2]),
                                    "low": float(k[3]),
                                    "close": float(k[4]),
                                    "volume": float(k[5])
                                })
                            current = int(klines[-1][0]) + 1
                        else:
                            print(f"Error en respuesta: {data}")
                            break
                        time.sleep(0.1)  # evitar rate-limit
                    except Exception as e:
                        print(f"Error obteniendo datos: {e}")
                        break

        if all_data:
            df = pd.DataFrame(all_data)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            # print(f"Datos obtenidos: {len(df)} velas")
            return df

        print("Sin datos devueltos")
        return None

    def get_bybit_data_public(
            self,
            symbol: str = "AVAXUSDT",
            interval: str = "5",
            start_date: str | None = "2025-06-01",
            end_date: str | None = "2025-06-02",
            limit: int = 0
    ):
        """
        Obtiene datos OHLC de la API pública Bybit.
        ▸ Si limit == 0  ⇒  modo *rango de fechas* (comportamiento original).
        ▸ Si limit >  0  ⇒  modo *look-back*  (últimas «limit» velas).
        """

        url = "https://api.bybit.com/v5/market/kline"
        max_limit = 1000  # tope permitido por Bybit
        all_data = []

        # ────────────────────────────
        # 1)  MODO LOOK-BACK (limit > 0)
        # ────────────────────────────
        if limit > 0:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, max_limit)
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if data["retCode"] == 0 and data["result"]["list"]:
                    klines = data["result"]["list"]
                    for k in klines:
                        all_data.append({
                            "timestamp": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        })
            except Exception as e:
                print(f"Error (look-back): {e}")

        # ────────────────────────────────
        # 2)  MODO RANGO DE FECHAS (limit == 0)
        # ────────────────────────────────
        else:
            print(f"Obteniendo datos de {symbol} (API pública) desde {start_date} hasta {end_date}…")
            if not start_date or not end_date:
                raise ValueError("Debes especificar start_date y end_date en modo rango")

            # Calcular número de velas según el intervalo
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            minutes_per_candle = int(interval)
            num_candles = int((end_ts - start_ts) / (minutes_per_candle * 60 * 1000))

            if num_candles <= max_limit:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": str(start_ts),
                    "end": str(end_ts),
                    "limit": str(num_candles)
                }
                try:
                    r = requests.get(url, params=params, timeout=10)
                    data = r.json()
                    if data["retCode"] == 0 and data["result"]["list"]:
                        klines = data["result"]["list"]
                        for k in klines:
                            all_data.append({
                                "timestamp": int(k[0]),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5])
                            })
                except Exception as e:
                    print(f"Error (rango simple): {e}")
            else:
                # Si no cabe en una sola petición, usar el bucle clásico
                current = start_ts
                while current < end_ts:
                    params = {
                        "category": "linear",
                        "symbol": symbol,
                        "interval": interval,
                        "start": str(current),
                        "limit": max_limit
                    }
                    try:
                        r = requests.get(url, params=params, timeout=10)
                        data = r.json()
                        if data["retCode"] == 0 and data["result"]["list"]:
                            klines = data["result"]["list"]
                            for k in klines:
                                all_data.append({
                                    "timestamp": int(k[0]),
                                    "open": float(k[1]),
                                    "high": float(k[2]),
                                    "low": float(k[3]),
                                    "close": float(k[4]),
                                    "volume": float(k[5])
                                })
                            current = int(klines[-1][0]) + 1
                        else:
                            break
                        time.sleep(0.1)  # evitar rate-limit
                    except Exception as e:
                        print(f"Error obteniendo datos: {e}")
                        break

        # ────────────────────────────────
        # Construir DataFrame final
        # ────────────────────────────────
        if all_data:
            df = pd.DataFrame(all_data)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            print(f"Datos obtenidos: {len(df)} velas")
            return df

        print("Sin datos devueltos")
        return None

    def get_bybit_data_smart(self, symbol="AVAXUSDT", interval="5", start_date="2025-06-01", end_date="2025-06-02"):
        """
        Método inteligente que maneja cache y descarga solo datos faltantes
        """
        filename = self.get_data_filename(symbol, interval, start_date, end_date)

        # Intentar cargar datos cacheados
        cached_df = self.load_cached_data(filename)

        if cached_df is not None:
            # Verificar si las fechas solicitadas están completamente cubiertas
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            cached_start = cached_df['datetime'].min()
            cached_end = cached_df['datetime'].max()

            if start_dt >= cached_start and end_dt <= cached_end:
                print("Todos los datos ya están en cache, no es necesario descargar")
                # Filtrar solo el rango solicitado
                mask = (cached_df['datetime'] >= start_dt) & (cached_df['datetime'] <= end_dt)
                return cached_df[mask].reset_index(drop=True)

        # Determinar qué rangos necesitamos descargar
        missing_ranges = self.get_missing_date_ranges(cached_df, start_date, end_date)

        if not missing_ranges:
            print("Todos los datos ya están disponibles en cache")
            return cached_df

        print(f"Necesario descargar {len(missing_ranges)} rangos de fechas faltantes")

        # Descargar datos faltantes
        new_data_list = []
        for start_missing, end_missing in missing_ranges:
            print(f"Descargando rango: {start_missing} a {end_missing}")

            # Usar API privada si está disponible, sino API pública
            if self.api_key and self.api_secret:
                new_data = self.get_bybit_data_private(symbol, interval, start_missing, end_missing)
            else:
                new_data = self.get_bybit_data_public(symbol, interval, start_missing, end_missing)

            if new_data is not None and len(new_data) > 0:
                new_data_list.append(new_data)

        # Combinar todos los datos
        all_dataframes = []
        if cached_df is not None:
            all_dataframes.append(cached_df)
        all_dataframes.extend(new_data_list)

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            # Eliminar duplicados basados en timestamp
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(
                drop=True)

            # Guardar datos actualizados en cache
            self.save_data_to_cache(combined_df, filename)

            # Filtrar solo el rango solicitado
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            mask = (combined_df['datetime'] >= start_dt) & (combined_df['datetime'] <= end_dt)
            final_df = combined_df[mask].reset_index(drop=True)

            print(f"Datos finales: {len(final_df)} registros para el período {start_date} - {end_date}")
            return final_df

        print("No se pudieron obtener datos")
        return None

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """
        Calcula el oscilador estocástico
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def add_indicators(self, df):
        """
        Añade todos los indicadores estocásticos, RSI, ATR y ADX al DataFrame,
        y clasifica la tendencia futura usando un umbral absoluto basado en ATR.
        """
        # Estocástico con período K=100
        k100_raw, _ = self.calculate_stochastic(df['high'], df['low'], df['close'], 100, 1)
        df['k100'] = k100_raw.rolling(window=1).mean()
        df['d100'] = df['k100'].rolling(window=3).mean()
        df['e100'] = df['d100'].rolling(window=57).mean()

        # Estocástico estándar
        df['k'], df['d'] = self.calculate_stochastic(df['high'], df['low'], df['close'], 14, 3)
        df['e'] = df['d'].rolling(window=40).mean()

        # ATR clásico (ya lo tienes, pero aseguramos consistencia)
        atr_period = 10
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=atr_period).mean()

        # Trend EMA
        for period in [10, 200]:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'trend_ema{period}'] = np.where(df['close'] > df[f'EMA_{period}'], 'up', 'down')


        # ADX (Average Directional Index) (14 periodos)
        adx_period = 14
        up_move = df['high'].diff()
        down_move = df['low'].diff() * -1

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat([
            (df['high'] - df['low']),
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=adx_period).mean()

        plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['ADX_14'] = dx.rolling(window=adx_period).mean()

        # --- d) Porcentaje de velas verdes/rojas en los últimos N periodos ---
        n_bars = 12  # Puedes ajustar el número de velas
        df['candle_green'] = (df['close'] > df['open']).astype(int)
        df['candle_red'] = (df['close'] < df['open']).astype(int)
        df['pct_green_last_n'] = df['candle_green'].rolling(window=n_bars).mean()
        df['pct_red_last_n'] = df['candle_red'].rolling(window=n_bars).mean()

        # --- e) Volumen promedio/cambio en los últimos N periodos ---
        df['avg_volume_last_n'] = df['volume'].rolling(window=n_bars).mean()
        df['pct_volume_change'] = df['volume'].pct_change(periods=n_bars)

        # VWAP Trend
        # VWAP tradicional: suma acumulada de (precio*vol) / suma acumulada de volumen
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['VWAP'] = vwap
        df[f'trend_vwap'] = np.where(df['close'] > df[f'VWAP'], 'up', 'down')

        # --- f) Tamaño de la vela actual ---
        df['candle_body_size'] = (df['close'] - df['open']).abs()
        df['candle_total_size'] = (df['high'] - df['low']).abs()

        # Dirección de e100 y e14
        df['e100_direction'] = np.where(df['e100'].diff() > 0, 'ascending', 'descending')
        df['e14_direction'] = np.where(df['e'].diff() > 0, 'ascending', 'descending')

        # Hora, minuto y dia, tiempo ciclico
        df['hour'] = df['datetime'].dt.hour  # Extraer la hora
        df['minute'] = df['datetime'].dt.minute  # Extraer el minuto
        df['time_combined'] = df['hour'] * 100 + df['minute']
        df['day_of_week'] = df['datetime'].dt.weekday
        df['minute_of_day'] = df['hour'] * 60 + df['minute']
        df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
        # Day of the week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['volatility'] = df['close'].pct_change().rolling(window=20).std()


        # New Features test
        eps = 1e-9  # para evitar divisiones por cero

        # Retornos (versiones seguras)
        df['return_1'] = np.log(df['close'] / (df['close'].shift(1) + eps))
        df['return_5'] = np.log(df['close'] / (df['close'].shift(5) + eps))
        df['hl_pct'] = (df['high'] - df['low']) / (np.abs(df['close']) + eps)
        df['ch_pct'] = (df['close'] - df['open']) / (np.abs(df['open']) + eps)

        # pct_change seguro (si t-1 es 0, evita inf)
        def safe_pct_change(s, periods=1):
            prev = s.shift(periods)
            return (s - prev) / (np.abs(prev) + eps)

        df['vol_chg'] = safe_pct_change(df['volume'], 1)
        df['roc_10'] = safe_pct_change(df['close'], 10)
        df['roc_30'] = safe_pct_change(df['close'], 30)

        # Volatilidad
        df['return_1_raw'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['volatility_10'] = df['return_1_raw'].rolling(10).std()
        df['volatility_50'] = df['return_1_raw'].rolling(50).std()
        df.drop(columns=['return_1_raw'], inplace=True)

        df['tr'] = np.maximum(df['high'] - df['low'],
                              np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                                         np.abs(df['low'] - df['close'].shift(1))))
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_norm'] = df['atr_14'] / (np.abs(df['close']) + eps)

        # Momentum
        df['momentum_diff'] = df['roc_10'] - df['roc_30']

        # RSI robusto
        def compute_rsi(series, window):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window).mean()
            ma_down = down.rolling(window).mean()
            rs = ma_up / (ma_down + eps)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df['rsi_14'] = compute_rsi(df['close'], 14)
        df['rsi_50'] = compute_rsi(df['close'], 50)

        # Tendencia
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_ratio'] = df['ema_10'] / (df['ema_50'] + eps)
        df['price_above_ema50'] = (df['close'] > df['ema_50']).astype(int)

        # Tendencia simple (siguiente vela)
        df['price_change'] = df['close'].shift(-1) - df['close']
        df['trend'] = np.where(df['price_change'] > 0, 'bullish',
                               np.where(df['price_change'] < 0, 'bearish', 'lateral'))
        df.loc[df['price_change'].isna(), 'trend'] = np.nan
        lookahead = self.lookahead

        df = self.assign_future_trend_with_dd_order_sensitive(
            df,
            lookahead=lookahead,
            atr_mult=self.atr_mult,
            atr_dd_mult=getattr(self, "atr_stop", 2.6),
            order_sensitive=getattr(self, "order_sensitive_dd", True),
            drop_helpers=False  # o False si quieres auditar targets/stops
        )

        # df = self.assign_future_trend_with_dd_first_event(
        #     df,
        #     lookahead=lookahead,
        #     atr_mult=self.atr_mult,
        #     atr_dd_mult=getattr(self, "atr_stop", 2.6),
        #     first_event=True,
        #     drop_helpers=False  # o False si quieres auditar targets/stops
        # )

        # Donde no hay futuro, asigna NaN a la tendencia futura

        return df

    def assign_future_trend_with_dd_first_event(self, df,
                                                lookahead=60,
                                                atr_mult=None,
                                                atr_dd_mult=1.0,
                                                first_event=True,  # Variante A activada por defecto
                                                drop_helpers=True):
        """
        Rotula 'future_trend' usando TP/SL por ATR.

        Variante A (first_event=True):
          - Recorre la ventana futura y decide por el PRIMER evento entre:
              TP_up, TP_down, stop_bull, stop_bear
            Reglas:
              * Si el primer evento es TP_up  -> 'bullish'
              * Si el primer evento es TP_down-> 'bearish'
              * Si el primer evento es un stop (de cualquier lado) -> 'lateral'
          - Las últimas 'lookahead' filas quedan en NaN (no hay futuro completo).

        Modo rápido (first_event=False):
          - 'bullish' si (alcanza TP_up) y NUNCA viola stop_bull en la ventana.
          - 'bearish' si (alcanza TP_down) y NUNCA viola stop_bear en la ventana.
          - Si ambas condiciones fallan -> 'lateral'.

        Requiere: columnas ['high','low','close','ATR'] en df.
        Crea/usa: 'target_up','target_down','stop_bull','stop_bear','future_trend'.
        """

        if atr_mult is None:
            atr_mult = getattr(self, "atr_mult", 1.0)

        L = int(lookahead)
        n = len(df)
        if n <= L:
            raise ValueError(f"DataFrame demasiado corto para lookahead={L} (n={n}).")

        # --- Targets (TP) y Stops (DD) ---
        df['target_up'] = df['close'] + df['ATR'] * float(atr_mult)
        df['target_down'] = df['close'] - df['ATR'] * float(atr_mult)
        df['stop_bull'] = df['close'] - df['ATR'] * float(atr_dd_mult)  # long pierde si cae bajo esto
        df['stop_bear'] = df['close'] + df['ATR'] * float(atr_dd_mult)  # short pierde si sube sobre esto

        # --- Ventanas futuras alineadas (vectorizadas) ---
        future_max_high = df['high'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_low = df['low'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))
        future_max_close = df['close'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_close = df['close'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))

        # Eventos agregados en la ventana (para el modo rápido y para detectar ambigüedades)
        bullish_tp = (future_max_high > df['target_up']) | (future_max_close > df['target_up'])
        bearish_tp = (future_min_low < df['target_down']) | (future_min_close < df['target_down'])
        bullish_dd_breach = (future_min_low < df['stop_bull']) | (future_min_close < df['stop_bull'])
        bearish_dd_breach = (future_max_high > df['stop_bear']) | (future_max_close > df['stop_bear'])

        if not first_event:
            # ---------- Modo rápido (sin orden, conservador) ----------
            bullish_valid = bullish_tp & ~bullish_dd_breach
            bearish_valid = bearish_tp & ~bearish_dd_breach

            df['future_trend'] = np.select(
                [bullish_valid, bearish_valid],
                ['bullish', 'bearish'],
                default='lateral'
            )
        else:
            # ---------- Variante A: primer evento gana ----------
            # Base rápida: si exactamente uno es válido, asigna sin escanear
            bullish_valid = (bullish_tp & ~bullish_dd_breach)
            bearish_valid = (bearish_tp & ~bearish_dd_breach)

            # Asignación base (unívoca)
            future_trend = np.where(bullish_valid, 'bullish',
                                    np.where(bearish_valid, 'bearish', 'lateral')).astype(object)

            # Necesitan escaneo las filas donde NO hay un único válido,
            # pero sí hay algún evento en la ventana:
            any_event = (bullish_tp | bearish_tp | bullish_dd_breach | bearish_dd_breach)
            # (A ^ B) = True si exactamente uno es True. Negamos para coger 0 o 2 verdaderos.
            need_scan = (~(bullish_valid ^ bearish_valid)) & any_event
            idxs = np.where(need_scan.fillna(False).values)[0]

            # Arrays para escanear por tiempo real
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            tgt_up = df['target_up'].values
            tgt_dn = df['target_down'].values
            st_bu = df['stop_bull'].values
            st_be = df['stop_bear'].values

            for i in idxs:
                start, end = i + 1, min(i + 1 + L, n)
                decided = False
                for t in range(start, end):
                    # 1) TPs primero (si ocurren antes, ganan al stop)
                    if (highs[t] > tgt_up[i]) or (closes[t] > tgt_up[i]):
                        future_trend[i] = 'bullish';
                        decided = True;
                        break
                    if (lows[t] < tgt_dn[i]) or (closes[t] < tgt_dn[i]):
                        future_trend[i] = 'bearish';
                        decided = True;
                        break
                    # 2) Si aún no hubo TP, cualquier stop ⇒ lateral
                    if (lows[t] < st_bu[i]) or (closes[t] < st_bu[i]) \
                            or (highs[t] > st_be[i]) or (closes[t] > st_be[i]):
                        future_trend[i] = 'lateral';
                        decided = True;
                        break
                if not decided:
                    future_trend[i] = 'lateral'

            df['future_trend'] = future_trend

        # Últimas L filas -> NaN (no hay futuro completo)
        df.loc[df.index[-L:], 'future_trend'] = np.nan

        # Limpieza opcional
        if drop_helpers:
            df.drop(columns=['target_up', 'target_down', 'stop_bull', 'stop_bear'],
                    inplace=True, errors='ignore')

        # Guarda lookahead para otras partes (train split, etc.)
        self.lookahead = L
        return df

    def assign_future_trend_with_dd_order_sensitive(self, df,
                                                    lookahead=25,
                                                    atr_mult=None,
                                                    atr_dd_mult=1.0,
                                                    order_sensitive=True,
                                                    drop_helpers=True):
        """
        Rotula 'future_trend' usando objetivos (TP) por ATR y un máximo drawdown (stop) por ATR.

        Reglas:
          - Bullish válido si alcanza target_up y NO viola stop_bull dentro de las próximas 'lookahead' velas.
          - Bearish válido si alcanza target_down y NO viola stop_bear dentro de las próximas 'lookahead' velas.
          - Si order_sensitive=True y en una fila se tocan ambos (TP y stop), decide por el evento que ocurre primero.
          - Las últimas 'lookahead' filas quedan en NaN (no hay futuro completo).

        Requiere en df: ['high','low','close','ATR'].
        Crea/usa: 'target_up','target_down','stop_bull','stop_bear','future_trend'.
        """
        if atr_mult is None:
            atr_mult = getattr(self, "atr_mult", 1.0)

        L = int(lookahead)

        # Targets (TP) y Stops (DD permitido)
        df['target_up'] = df['close'] + df['ATR'] * float(atr_mult)
        df['target_down'] = df['close'] - df['ATR'] * float(atr_mult)
        df['stop_bull'] = df['close'] - df['ATR'] * float(atr_dd_mult)  # long pierde si cae bajo esto
        df['stop_bear'] = df['close'] + df['ATR'] * float(atr_dd_mult)  # short pierde si sube sobre esto

        # Ventanas futuras alineadas
        future_max_high = df['high'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_low = df['low'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))
        future_max_close = df['close'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_close = df['close'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))

        # TP hits (como tenías)
        bullish_tp = (future_max_high >= df['target_up']) | (future_max_close >= df['target_up'])
        bearish_tp = (future_min_low <= df['target_down']) | (future_min_close <= df['target_down'])

        # Breach de stop / DD
        bullish_dd_breach = (future_min_low <= df['stop_bull']) | (future_min_close <= df['stop_bull'])
        bearish_dd_breach = (future_max_high >= df['stop_bear']) | (future_max_close >= df['stop_bear'])

        if not order_sensitive:
            # Modo simple (rápido y conservador)
            bullish_valid = bullish_tp & ~bullish_dd_breach
            bearish_valid = bearish_tp & ~bearish_dd_breach

            df['future_trend'] = np.select(
                [bullish_valid, bearish_valid],
                ['bullish', 'bearish'],
                default='lateral'
            )
        else:
            # Modo sensible al orden: resolver conflictos (TP y DD verdaderos a la vez)
            bullish_valid = bullish_tp & ~bullish_dd_breach
            bearish_valid = bearish_tp & ~bearish_dd_breach

            # Base
            future_trend = np.where(bullish_valid, 'bullish',
                                    np.where(bearish_valid, 'bearish', 'lateral')).astype(object)

            conflict_bull = (bullish_tp & bullish_dd_breach).fillna(False).values
            conflict_bear = (bearish_tp & bearish_dd_breach).fillna(False).values

            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            tgt_up = df['target_up'].values
            st_bu = df['stop_bull'].values
            tgt_dn = df['target_down'].values
            st_be = df['stop_bear'].values
            n = len(df)

            # Solo recorremos índices conflictivos (pocos)
            idx_conf_bull = np.where(conflict_bull)[0]
            for i in idx_conf_bull:
                start, end = i + 1, min(i + 1 + L, n)
                first = None
                for t in range(start, end):
                    if (highs[t] >= tgt_up[i]) or (closes[t] >= tgt_up[i]):
                        first = 'tp';
                        break
                    if (lows[t] <= st_bu[i]) or (closes[t] <= st_bu[i]):
                        first = 'sl';
                        break
                future_trend[i] = 'bullish' if first == 'tp' else ('lateral' if first is None else 'lateral')

            idx_conf_bear = np.where(conflict_bear)[0]
            for i in idx_conf_bear:
                start, end = i + 1, min(i + 1 + L, n)
                first = None
                for t in range(start, end):
                    if (lows[t] <= tgt_dn[i]) or (closes[t] <= tgt_dn[i]):
                        first = 'tp';
                        break
                    if (highs[t] >= st_be[i]) or (closes[t] >= st_be[i]):
                        first = 'sl';
                        break
                future_trend[i] = 'bearish' if first == 'tp' else ('lateral' if first is None else 'lateral')

            df['future_trend'] = future_trend

        # Las últimas L filas no tienen futuro completo → NaN
        df.loc[df.index[-L:], 'future_trend'] = np.nan

        # Limpieza opcional
        if drop_helpers:
            df.drop(columns=[
                'target_up', 'target_down', 'stop_bull', 'stop_bear'
            ], inplace=True, errors='ignore')

        # Guarda lookahead en la instancia (útil para excluir en training)
        self.lookahead = L
        return df

    def add_volatility_zones(self, df):
        # Puedes ajustar percentiles o usar valores absolutos si prefieres
        low_vol = df['volatility'].quantile(0.33)
        high_vol = df['volatility'].quantile(0.66)
        df['vol_zone'] = np.select(
            [
                df['volatility'] < low_vol,
                (df['volatility'] >= low_vol) & (df['volatility'] < high_vol),
                df['volatility'] >= high_vol
            ],
            ['low', 'medium', 'high'],
            default='unknown'
        )
        return df

    def analyze_data_quality(self, df):
        """
        Analiza la calidad de los datos y distribución de tendencias
        """
        print("=== ANÁLISIS DE CALIDAD DE DATOS ===")
        print(f"Total de registros: {len(df)}")
        print(f"Registros con future_trend válido: {df['future_trend'].notna().sum()}")

        # Distribución de tendencias
        trend_distribution = df['future_trend'].value_counts()
        print("\n=== DISTRIBUCIÓN DE TENDENCIAS ===")
        for trend, count in trend_distribution.items():
            percentage = (count / len(df)) * 100
            print(f"{trend}: {count} ({percentage:.2f}%)")

        # Estadísticas de price_change_5
        # print("\n=== ESTADÍSTICAS DE CAMBIO DE PRECIO (5 períodos) ===")
        # print(f"Media: {df['price_change_lookahead'].mean():.6f}")
        # print(f"Mediana: {df['price_change_lookahead'].median():.6f}")
        # print(f"Desviación estándar: {df['price_change_lookahead'].std():.6f}")
        # print(f"Min: {df['price_change_lookahead'].min():.6f}")
        # print(f"Max: {df['price_change_lookahead'].max():.6f}")

    def create_combinations(self, df):
        """
        Crea combinaciones mejoradas con zonas específicas y relaciones posicionales
        """
        combinations = []

        # Definir zonas específicas para cada indicador
        zones = {
            'oversold_extreme': (0, 10),  # Sobreventa extrema
            'oversold': (10, 20),  # Sobreventa
            'low': (20, 30),  # Bajo
            'low_neutral': (30, 40),  # Bajo-neutral
            'neutral_low': (40, 50),  # Neutral-bajo
            'neutral_high': (50, 60),  # Neutral-alto
            'high': (60, 70),  # Alto
            'high_neutral': (70, 80),  # Alto-neutral
            'overbought': (80, 90),  # Sobrecompra
            'overbought_extreme': (90, 100)  # Sobrecompra extrema
        }

        # # Relaciones posicionales específicas
        # stoch14_relations = ['k>d>e', 'e>d>k', 'any']  # Reducir relaciones
        # stoch100_relations = ['k>d>e', 'e>d>k', 'any']  # Reducir relaciones
        #
        # # Direcciones de tendencia para E100
        # directions_e100 = ['ascending', 'descending']
        # directions_e14 = ['ascending', 'descending']

        # Generar combinaciones específicas con zonas y relaciones
        # for k14_zone_name, k14_zone in zones.items():
        #     for d14_zone_name, d14_zone in zones.items():
        #         for e14_zone_name, e14_zone in zones.items():
        #             for k100_zone_name, k100_zone in zones.items():
        #                 for d100_zone_name, d100_zone in zones.items():
        #                     for e100_zone_name, e100_zone in zones.items():
        #                         for stoch14_rel in stoch14_relations:
        #                             for stoch100_rel in stoch100_relations:
        #                                 for direction_e100 in directions_e100:
        #                                     for direction_e14 in directions_e14:
        #                                         combination = {
        #                                             # Zonas específicas para estocástico 14
        #                                             'k14_zone': k14_zone_name,
        #                                             'k14_min': k14_zone[0],
        #                                             'k14_max': k14_zone[1],
        #                                             'd14_zone': d14_zone_name,
        #                                             'd14_min': d14_zone[0],
        #                                             'd14_max': d14_zone[1],
        #                                             'e14_zone': e14_zone_name,
        #                                             'e14_min': e14_zone[0],
        #                                             'e14_max': e14_zone[1],
        #
        #                                             # Zonas específicas para estocástico 100
        #                                             'k100_zone': k100_zone_name,
        #                                             'k100_min': k100_zone[0],
        #                                             'k100_max': k100_zone[1],
        #                                             'd100_zone': d100_zone_name,
        #                                             'd100_min': d100_zone[0],
        #                                             'd100_max': d100_zone[1],
        #                                             'e100_zone': e100_zone_name,
        #                                             'e100_min': e100_zone[0],
        #                                             'e100_max': e100_zone[1],
        #
        #                                             # Relaciones posicionales
        #                                             'stoch14_relation': stoch14_rel,
        #                                             'stoch100_relation': stoch100_rel,
        #
        #                                             # Dirección de E100
        #                                             'e100_direction': direction_e100,
        #                                             'e14_direction': direction_e14,
        #
        #                                             'type': 'zone_positional'
        #                                         }
        #                                         combinations.append(combination)
        e14_zones = list(zones.items())
        e100_zones = list(zones.items())
        d100_zones = list(zones.items())

        stoch14_relations = ['k>d>e', 'k<d<e']
        stoch100_relations = ['k>d>e', 'k<d<e']
        directions = ['ascending', 'descending']
        combinations = []
        vol_zones = ['low', 'medium', 'high']
        for e14_name, e14_range in e14_zones:
            for e100_name, e100_range in e100_zones:
                for d100_name, d100_range in d100_zones:
                    for rel14 in stoch14_relations:
                        for rel100 in stoch100_relations:
                            for dir_e14 in directions:
                                for dir_e100 in directions:
                                    for vol_zone in vol_zones:
                                        combinations.append({
                                            'e14_zone': e14_name,
                                            'e14_min': e14_range[0],
                                            'e14_max': e14_range[1],
                                            'e100_zone': e100_name,
                                            'e100_min': e100_range[0],
                                            'e100_max': e100_range[1],
                                            'd100_zone': d100_name,
                                            'd100_min': d100_range[0],
                                            'd100_max': d100_range[1],
                                            'stoch14_relation': rel14,
                                            'stoch100_relation': rel100,
                                            'e14_direction': dir_e14,
                                            'e100_direction': dir_e100,
                                            'vol_zone': vol_zone,  # <-- Nuevo campo
                                            'type': 'simplified'
                                        })
        print(f"Generadas {len(combinations)} combinaciones simplificadas (con d100, e100 y volatilidad)")
        return combinations

    def analyze_combination(self, df, combination):
        if combination.get('type') == 'simplified':
            mask_e14 = (df['e'] >= combination['e14_min']) & (df['e'] < combination['e14_max'])
            mask_e100 = (df['e100'] >= combination['e100_min']) & (df['e100'] < combination['e100_max'])
            mask_d100 = (df['d100'] >= combination['d100_min']) & (df['d100'] < combination['d100_max'])
            mask_e14_dir = (df['e14_direction'] == combination['e14_direction'])
            mask_e100_dir = (df['e100_direction'] == combination['e100_direction'])

            # Relación stoch14
            if combination['stoch14_relation'] == 'k>d>e':
                mask_rel14 = (df['k'] > df['d']) & (df['d'] > df['e'])
            elif combination['stoch14_relation'] == 'k<d<e':
                mask_rel14 = (df['k'] < df['d']) & (df['d'] < df['e'])
            else:
                mask_rel14 = pd.Series([True] * len(df), index=df.index)

            # Relación stoch100
            if combination['stoch100_relation'] == 'k>d>e':
                mask_rel100 = (df['k100'] > df['d100']) & (df['d100'] > df['e100'])
            elif combination['stoch100_relation'] == 'k<d<e':
                mask_rel100 = (df['k100'] < df['d100']) & (df['d100'] < df['e100'])
            else:
                mask_rel100 = pd.Series([True] * len(df), index=df.index)

            mask_vol = (df['vol_zone'] == combination['vol_zone'])
            mask = mask_e14 & mask_e100 & mask_d100 & mask_e14_dir & mask_e100_dir & mask_vol & mask_rel14 & mask_rel100 & \
                   df['future_trend'].notna()

            filtered_data = df[mask]
            if len(filtered_data) < 10:
                return None

            total_samples = len(filtered_data)
            bullish_count = (filtered_data['future_trend'] == 'bullish').sum()
            bearish_count = (filtered_data['future_trend'] == 'bearish').sum()
            lateral_count = (filtered_data['future_trend'] == 'lateral').sum()

            bullish_pct = bullish_count / total_samples * 100
            bearish_pct = bearish_count / total_samples * 100
            lateral_pct = lateral_count / total_samples * 100

            if bullish_pct > 50:
                dominant_trend = 'bullish'
                confidence = bullish_pct
            elif bearish_pct > 50:
                dominant_trend = 'bearish'
                confidence = bearish_pct
            elif lateral_pct > 40:
                dominant_trend = 'lateral'
                confidence = lateral_pct
            else:
                dominant_trend = 'mixed'
                confidence = max(bullish_pct, bearish_pct, lateral_pct)

            result = combination.copy()
            result.update({
                'total_samples': total_samples,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'lateral_count': lateral_count,
                'bullish_pct': bullish_pct,
                'bearish_pct': bearish_pct,
                'lateral_pct': lateral_pct,
                'dominant_trend': dominant_trend,
                'confidence': confidence
            })
            return result
        else:
            return None

    def run_analysis(self, symbol="AVAXUSDT", start_date="2025-06-01", end_date="2025-06-02", interval="5"):
        """
        Ejecuta el análisis completo con cache inteligente
        """
        # Obtener datos usando cache inteligente
        self.data = self.get_bybit_data_smart(symbol, interval, start_date, end_date)

        if self.data is None or len(self.data) == 0:
            print("No se pudieron obtener datos")
            return

        # Añadir indicadores
        self.data = self.add_indicators(self.data)

        self.data = self.add_volatility_zones(self.data)

        # Analizar calidad de datos
        self.analyze_data_quality(self.data)

        # # Crear combinaciones
        # combinations = self.create_combinations(self.data)
        # print(f"\nAnalizando {len(combinations)} combinaciones...")
        #
        # # Analizar cada combinación
        # self.results = []
        # for i, combination in enumerate(combinations):
        #     if i % 1000 == 0:
        #         print(f"Progreso: {i}/{len(combinations)}")
        #
        #     result = self.analyze_combination(self.data, combination)
        #     if result:
        #         self.results.append(result)
        #
        # print(f"Análisis completado. {len(self.results)} combinaciones válidas encontradas.")

    def save_results(self, filename="stochastic_analysis_results.csv"):
        """
        Guarda los resultados en CSV incluyendo información de zonas
        """
        if not self.results:
            print("No hay resultados para guardar")
            return

        df_results = pd.DataFrame(self.results)

        # Descripción legible para todos los casos simplificados
        if 'type' in df_results.columns:
            df_results['pattern_description'] = df_results.apply(
                lambda row: self.create_pattern_description(row) if row.get('type') in ('zone_positional', 'simplified')
                else 'legacy_pattern', axis=1
            )

        df_results = df_results.sort_values('confidence', ascending=False)
        df_results.to_csv(filename, index=False)
        print(f"Resultados guardados en {filename}")

        return df_results

    def create_pattern_description(self, row):
        return (f"E14:{row['e14_zone']}({row['e14_direction']}) | "
                f"E100:{row['e100_zone']}({row['e100_direction']}) | "
                f"D100:{row['d100_zone']} | "
                f"Stoch14:{row.get('stoch14_relation', '?')} | "
                f"Stoch100:{row.get('stoch100_relation', '?')} | "
                f"Vol:{row.get('vol_zone', '?')}")

    def fit_threshold_curve(
            self,
            X_hold,
            y_hold,
            u_vals=None,
            already_scaled=True,
            features=None,
            # === Parámetros Opción A (data-driven) ===
            perc_low=0.40,  # percentil inferior para top1 (en casos de tendencia)
            perc_high=0.90,  # percentil superior para top1 (en casos de tendencia)
            floor=0.50,  # no bajar de 0.50
            ceil=0.70,  # cap base; si None, sin cap base
            step=0.01,  # resolución del grid
            # === Mínimo dinámico de muestras de tendencia ===
            min_trend_samples=None,  # si None, se calculará dinámicamente
            min_abs=300,  # piso absoluto
            min_frac_total=0.03,  # 3% del holdout total
            min_frac_trend=0.10,  # 10% de las muestras de tendencia
            cap_trend_samples=2000,  # tope para no exigir demasiado
            # === Cap dinámico de la cola ===
            tail_min_count=200,  # mínimo absoluto de muestras sobre ceil para liberar el cap
            tail_min_frac=0.02,  # o mínimo % relativo sobre n_trend
            ceil_max=0.90,  # techo máximo cuando se libera el cap
            # === NUEVO: guardas para evitar umbrales demasiado bajos ===
            hard_floor_u=None,  # ej: 0.60 -> descarta u<0.60
            precision_guard=None  # ej: 0.80 -> corta curva por debajo del 1er u con prec>=0.80
    ):
        """
        Calcula trade_rate y precision para umbrales 'u' (top1 >= u y clase top ∈ {bullish,bearish}).
        Si u_vals es None, construye el rango automáticamente con percentiles de top1 en muestras de tendencia.
        Aplica cap dinámico: si hay suficiente masa de probas > ceil, permite subir el tope hasta ceil_max.
        Devuelve un DataFrame con columnas: ['u','trade_rate','precision','n'] y metadatos en df.attrs.
        """

        # === Validación & coerción de X_hold ===
        if X_hold is None:
            raise TypeError("X_hold es None. Pásame el holdout de features.")

        if not already_scaled:
            if features is None:
                features = getattr(self, "features_", None)
                if features is None:
                    raise ValueError("Debes pasar 'features' si already_scaled=False.")
            if hasattr(X_hold, "values"):  # DataFrame/Series
                X_arr = np.asarray(X_hold[features].values if hasattr(X_hold, "__getitem__") else X_hold.values)
            else:
                X_arr = np.asarray(X_hold)
            X_arr = self.scaler.transform(X_arr)
        else:
            X_arr = np.asarray(X_hold.values) if hasattr(X_hold, "values") else np.asarray(X_hold)

        if X_arr.ndim != 2:
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            else:
                raise TypeError(
                    f"Se esperaba array 2D, recibido ndim={X_arr.ndim}, shape={getattr(X_arr, 'shape', None)}")

        # === Modelo listo ===
        mdl = getattr(self, "model", None)
        if mdl is None or not hasattr(mdl, "predict_proba"):
            raise RuntimeError("self.model no está entrenado o no soporta predict_proba().")

        # === Probabilidades y clases top ===
        proba = mdl.predict_proba(X_arr)
        classes = list(mdl.classes_)
        top = proba.max(axis=1)
        top_idx = proba.argmax(axis=1)
        cls_top = np.array([classes[i] for i in top_idx])
        is_trend = np.isin(cls_top, ["bullish", "bearish"])
        top_trend = top[is_trend]

        # === Mínimo dinámico de muestras de tendencia ===
        n_total = int(X_arr.shape[0])
        n_trend = int(top_trend.size)
        if min_trend_samples is None:
            min_trend_samples = max(
                int(min_abs),
                int(min_frac_total * n_total),
                int(min_frac_trend * n_trend)
            )
            min_trend_samples = min(int(min_trend_samples), int(cap_trend_samples))

        # Variables de auditoría
        q_lo = q_hi = None
        tail_count = tail_frac = None
        dynamic_cap_applied = False

        # === Construcción data-driven del rango de umbrales ===
        if u_vals is None:
            if n_trend >= min_trend_samples:
                q_lo, q_hi = np.quantile(top_trend, [perc_low, perc_high])
                u_min = max(float(floor), float(q_lo))

                if ceil is None:
                    # Sin cap base: usa q_hi directamente
                    u_max = float(q_hi)
                else:
                    # Cap base
                    u_max = min(float(ceil), float(q_hi))
                    # ¿Hay suficiente masa en la cola por encima del cap base?
                    if float(q_hi) > float(ceil):
                        tail_count = int((top_trend >= float(ceil)).sum())
                        tail_frac = tail_count / max(1, n_trend)
                        if (tail_count >= int(tail_min_count)) or (tail_frac >= float(tail_min_frac)):
                            # Libera el cap hasta ceil_max o q_hi
                            u_max = min(float(ceil_max), float(q_hi))
                            dynamic_cap_applied = True

                # Si el rango queda degenerado o no finito: fallback
                if (not np.isfinite(u_min)) or (not np.isfinite(u_max)) or (u_max <= u_min):
                    u_min, u_max = float(floor), min(float(ceil) if ceil is not None else float(floor) + 0.20,
                                                     float(floor) + 0.12)
            else:
                # Fallback seguro si no alcanza el mínimo
                u_min, u_max = float(floor), min(float(ceil) if ceil is not None else float(floor) + 0.20,
                                                 float(floor) + 0.12)

            # Grid de umbrales
            u_vals = np.round(np.arange(u_min, u_max + 1e-9, float(step)), 3)
            if u_vals.size < 3:
                u_vals = np.unique(
                    np.round(np.linspace(u_min, max(u_min + 2 * step, u_max), num=3), 3)
                )

        # === Métrica por umbral ===
        rows = []
        y_hold_arr = np.asarray(y_hold)
        for u in u_vals:
            m = (top >= u) & is_trend
            n = int(m.sum())
            if n == 0:
                rows.append((float(u), 0.0, float("nan"), 0))
                continue
            prec = float((y_hold_arr[m] == cls_top[m]).mean())
            trade_rate = n / len(y_hold_arr)
            rows.append((float(u), float(trade_rate), float(prec), n))

        df = pd.DataFrame(rows, columns=["u", "trade_rate", "precision", "n"])

        # === NUEVO: aplicar guardas ===
        # 1) Filtro por suelo duro de u
        if hard_floor_u is not None:
            hard_floor_u = float(hard_floor_u)
            df = df[df["u"] >= hard_floor_u].reset_index(drop=True)

        # 2) Corte por precisión mínima "primera vez que alcanza"
        #    Encuentra el menor u con precision >= precision_guard y descarta u por debajo.
        if precision_guard is not None:
            precision_guard = float(precision_guard)
            df_ok = df[df["precision"] >= precision_guard]
            if not df_ok.empty:
                u_guard = float(df_ok["u"].min())
                df = df[df["u"] >= u_guard].reset_index(drop=True)
                df.attrs["precision_guard_used"] = precision_guard
                df.attrs["u_guard_cut"] = u_guard

        # --- Metadatos (lo que ya tenías) ---
        df.attrs["u_vals"] = u_vals
        df.attrs["n_total"] = n_total
        df.attrs["n_trend"] = n_trend
        df.attrs["min_trend_samples_used"] = int(min_trend_samples)
        df.attrs["range_built_from_percentiles"] = bool(n_trend >= min_trend_samples)
        df.attrs["q_lo"] = None if q_lo is None else float(q_lo)
        df.attrs["q_hi"] = None if q_hi is None else float(q_hi)
        df.attrs["tail_count_over_ceil"] = tail_count
        df.attrs["tail_frac_over_ceil"] = tail_frac
        df.attrs["dynamic_cap_applied"] = bool(dynamic_cap_applied)
        if u_vals is not None and len(u_vals) > 0:
            # actualiza u_min/u_max tras guardas
            if not df.empty:
                df.attrs["u_min"] = float(df["u"].min())
                df.attrs["u_max"] = float(df["u"].max())
            else:
                df.attrs["u_min"] = float(u_vals.min())
                df.attrs["u_max"] = float(u_vals.max())

        return df

    def create_predictive_model(self, display_graphic=False, N_KEEP=40000):
        if self.data is None:
            print("No hay datos para entrenar el modelo")
            return

        df_model, features = self.preparar_features(self.data)

        L = self.lookahead  # o fija 25 si prefieres

        n_total = len(df_model)
        if n_total <= L:
            raise RuntimeError(f"No hay suficientes filas para excluir lookahead={L} (n={n_total}).")

        # Si hay más de 40k filas, recorta a las últimas 40k
        if n_total > N_KEEP:
            df_model = df_model.iloc[-N_KEEP:].copy()
            n_total = len(df_model)  # ahora n_total == N_KEEP

        # Verificación: tras excluir L debe quedar algo de train
        if n_total - L < 1000:
            # umbral a tu gusto; evita entrenar con muy pocas filas
            print(f"Advertencia: solo {n_total - L} filas de train tras recorte y lookahead. "
                  f"Considera reducir L o N_KEEP.")

        # === Excluir las últimas L velas (lookahead) del entrenamiento ===
        df_train = df_model.iloc[:-L].copy()

        # Si ya marcaste NaN en future_trend para las últimas L, bastaría con notna();
        # aquí hacemos ambas cosas por seguridad:
        df_train = df_train[df_train["future_trend"].notna()]

        # Quita filas con NaN en features (por rolling, etc.)
        df_train = df_train.dropna(subset=features + ["future_trend"])

        # (Opcional) guarda las últimas L para auditoría / live
        # self._holdout_future_window = df_model.iloc[-L:].copy()

        X = df_train[features]
        y = df_train["future_trend"]

        # === Split, escalado y entrenamiento ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1) split para calibración (no uses el test!)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        # 2) entrena una sola vez
        def expected_calibration_error(y_true, y_prob, n_bins=10):
            """ECE clásico sobre la confianza máxima."""
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            conf_max = np.max(y_prob, axis=1)
            pred_cls = np.argmax(y_prob, axis=1)
            # Mapear y_true a índices 0..K-1 según el orden de clases que uses afuera
            return_pred = None  # placeholder para consistencia, no se usa

        def multiclass_brier(y_true, y_prob, classes):
            """
            Brier multiclase: promedio sobre muestras de sum_k (p_k - y_k)^2.
            y_true: iterable de labels (no one-hot)
            y_prob: ndarray shape (n_samples, n_classes)
            classes: iterable con el orden de columnas de y_prob (model.classes_)
            """
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_idx = np.array([class_to_idx[y] for y in y_true])
            Y = np.zeros_like(y_prob)
            Y[np.arange(len(y_true)), y_idx] = 1.0
            return float(np.mean(np.sum((y_prob - Y) ** 2, axis=1)))

        def ece_maxprob(y_true, y_prob, classes, n_bins=10):
            """ECE usando la confianza máxima."""
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_idx = np.array([class_to_idx[y] for y in y_true])

            conf = np.max(y_prob, axis=1)
            pred = np.argmax(y_prob, axis=1)

            bins = np.linspace(0.0, 1.0, n_bins + 1)
            binids = np.digitize(conf, bins) - 1

            ece = 0.0
            n = len(y_true)
            for i in range(n_bins):
                mask = binids == i
                if np.any(mask):
                    acc = np.mean(pred[mask] == y_idx[mask])
                    avg_conf = np.mean(conf[mask])
                    ece += (np.sum(mask) / n) * abs(acc - avg_conf)
            return float(ece)

        # ===============================
        # Entrenamiento modelo base
        # ===============================
        base_model = RandomForestClassifier(
            n_estimators=700, max_features="sqrt", n_jobs=-1, random_state=42
        )
        base_model.fit(X_tr, y_tr)

        # ==== Sin calibrar ====
        preds_vanilla = base_model.predict(X_test_scaled)
        proba_vanilla = base_model.predict_proba(X_test_scaled)
        classes = base_model.classes_

        print("Reporte de clasificación de modelo SIN calibrar:")
        print(classification_report(y_test, preds_vanilla))
        print(f"Brier (multiclase): {multiclass_brier(y_test, proba_vanilla, classes):.6f}")
        print(f"Log loss: {log_loss(y_test, proba_vanilla, labels=classes):.6f}")
        print(f"ECE: {ece_maxprob(y_test, proba_vanilla, classes):.6f}")
        print("=" * 50)

        # ==== Con calibración ====
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
        self.model.fit(X_cal, y_cal)

        preds_cal = self.model.predict(X_test_scaled)
        proba_cal = self.model.predict_proba(X_test_scaled)
        classes_cal = self.model.classes_  # puede coincidir, pero toma el del calibrado

        print("Reporte de clasificación de modelo CALIBRADO:")
        print(classification_report(y_test, preds_cal))
        print(f"Brier (multiclase): {multiclass_brier(y_test, proba_cal, classes_cal):.6f}")
        print(f"Log loss: {log_loss(y_test, proba_cal, labels=classes_cal):.6f}")
        print(f"ECE: {ece_maxprob(y_test, proba_cal, classes_cal):.6f}")

        u_strong, info_strong, u_soft, info_soft, df_bands = self.select_thresholds_by_band_precision(
            X_hold=X_test_scaled, y_hold=y_test,
            band_step=0.1, u_min=0.55, u_max=0.95,
            min_margin=0.03,
            min_n_soft=1000,
            min_n_strong=1200,
            min_trade_rate_soft=0.10,
            min_trade_rate_strong=0.03,
            soft_range=(0.83, 0.865),
            strong_range=(0.89, 0.96),
            soft_start=0.60,
            strong_start=0.80,
            use_ci_lower=True,
            prefer_center_soft=True
        )

        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': base_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nImportancia de características:")
        print(feature_importance)

        print("u_strong:", u_strong, "| u_soft:", u_soft)
        print("STRONG:", info_strong)
        print("SOFT:", info_soft)

        # === Curvas de calibración por clase ===
        if display_graphic:
            proba_base = base_model.predict_proba(X_test_scaled)
            proba_calib = self.model.predict_proba(X_test_scaled)
            clases = base_model.classes_

            plt.figure(figsize=(7, 5))
            for target_class in clases:
                idx = list(clases).index(target_class)
                y_true_bin = np.array([1 if yt == target_class else 0 for yt in y_test])

                # Modelo sin calibrar
                prob_pred_base, prob_true_base = calibration_curve(
                    y_true_bin, proba_base[:, idx], n_bins=10, strategy='uniform'
                )
                plt.plot(prob_pred_base, prob_true_base, marker='o', label=f"Sin calibrar '{target_class}'")

                # # Modelo calibrado
                prob_pred_cal, prob_true_cal = calibration_curve(
                    y_true_bin, proba_calib[:, idx], n_bins=10, strategy='uniform'
                )
                plt.plot(prob_pred_cal, prob_true_cal, marker='o', linestyle='--', label=f"Calibrado '{target_class}'")

            fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
            plt.xlabel('Probabilidad predicha por el modelo')
            plt.ylabel('Proporción real de acierto')
            plt.title(f'Gráfico de calibración: sin calibrar vs calibrado\nEntrenado: {fecha_actual}')
            plt.legend()
            plt.grid()
            plt.show()

        return self.model

    def get_best_calibration_model(self, display_graphic=False,
                                   methods=("isotonic", "sigmoid", "beta", "histbin")):

        # ── Utilidades ──────────────────────────────────────────────────────────────
        def expected_calibration_error(y_true, proba, n_bins=15):
            """
            ECE multiclass (one-vs-rest promedio macro).
            y_true: Series/array de etiquetas (string/int)
            proba: ndarray [n_samples, n_classes] con predict_proba
            """
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            eces = []
            for k in range(len(classes)):
                p = proba[:, k]
                t = y_true_bin[:, k]
                # bins uniformes en probabilidad predicha
                bins = np.linspace(0, 1, n_bins + 1)
                ece = 0.0
                for i in range(n_bins):
                    mask = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
                    if np.any(mask):
                        conf = p[mask].mean()
                        acc = t[mask].mean()
                        w = mask.mean()
                        ece += w * abs(acc - conf)
                eces.append(ece)
            return float(np.mean(eces))

        class HistogramBinningCalibrator:
            """
            Calibrador simple por binning en cuantiles (multiclase via one-vs-rest).
            Regularización: mezcla con la frecuencia global de la clase (alpha).
            """

            def __init__(self, n_bins=20, alpha=0.2, quantile=True):
                self.n_bins = n_bins
                self.alpha = alpha
                self.quantile = quantile
                self.classes_ = None
                self.bin_edges_ = {}
                self.bin_means_ = {}
                self.class_priors_ = {}

            def fit(self, y_true, proba):
                self.classes_ = np.arange(proba.shape[1])
                y_true = np.asarray(y_true)
                for k in self.classes_:
                    p = proba[:, k]
                    t = (y_true == k).astype(int)
                    prior = t.mean()
                    self.class_priors_[k] = prior
                    if self.quantile:
                        edges = np.quantile(p, np.linspace(0, 1, self.n_bins + 1))
                        edges[0], edges[-1] = 0.0, 1.0
                    else:
                        edges = np.linspace(0, 1, self.n_bins + 1)
                    self.bin_edges_[k] = edges
                    means = []
                    for i in range(self.n_bins):
                        mask = (p >= edges[i]) & (p < edges[i + 1]) if i < self.n_bins - 1 else (p >= edges[i]) & (
                                p <= edges[i + 1])
                        if np.any(mask):
                            emp = t[mask].mean()
                            # mezcla con prior para suavizar
                            cal = (1 - self.alpha) * emp + self.alpha * prior
                        else:
                            cal = prior
                        means.append(cal)
                    self.bin_means_[k] = np.array(means, dtype=float)
                return self

            def predict_proba(self, proba):
                out = np.zeros_like(proba, dtype=float)
                for k in self.classes_:
                    p = proba[:, k]
                    edges = self.bin_edges_[k]
                    means = self.bin_means_[k]
                    idx = np.searchsorted(edges, p, side="right") - 1
                    idx = np.clip(idx, 0, len(means) - 1)
                    out[:, k] = means[idx]
                # renormaliza por fila para mantener suma=1 (suave)
                row_sum = out.sum(axis=1, keepdims=True)
                # Evita división por cero: mezcla con proba original si suma casi cero
                mask = row_sum.squeeze() > 1e-12
                out[mask] /= row_sum[mask]
                out[~mask] = proba[~mask]
                return out

        def fit_beta_calibration(y, proba):
            try:
                from betacal import BetaCalibration
            except Exception:
                return None  # no instalado

            classes = np.unique(y)
            y_bin = label_binarize(y, classes=classes)
            models = []

            for k in range(len(classes)):
                bc = BetaCalibration(parameters="abm")  # "abm" suele rendir bien
                bc.fit(proba[:, k].reshape(-1, 1), y_bin[:, k])
                models.append(bc)

            def _predict_one(bc, x):
                # Algunas versiones traen predict_proba; otras solo predict
                if hasattr(bc, "predict_proba"):
                    return bc.predict_proba(x).ravel()
                else:
                    return bc.predict(x).ravel()

            def predict_proba_fn(P):
                out = np.zeros_like(P, dtype=float)
                for k, bc in enumerate(models):
                    out[:, k] = _predict_one(bc, P[:, k].reshape(-1, 1))
                # renormaliza por fila para que sumen 1
                s = out.sum(axis=1, keepdims=True)
                out = np.divide(out, np.maximum(s, 1e-12))
                return out

            return predict_proba_fn

        # ── Tu flujo original (con mejoras) ────────────────────────────────────────
        if self.data is None:
            print("No hay datos para entrenar el modelo")
            return

        df_model, features = self.preparar_features(self.data)

        L = self.lookahead
        if len(df_model) <= L:
            raise RuntimeError(f"No hay suficientes filas para excluir lookahead={L}.")

        df_train = df_model.iloc[:-L].copy()
        df_train = df_train[df_train["future_trend"].notna()]
        df_train = df_train.dropna(subset=features + ["future_trend"])

        X = df_train[features]
        y = df_train["future_trend"]

        # Split principal
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Split interno para calibración (del train; no tocar test)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        # Entrena RF base una sola vez (prefit)
        base_model = RandomForestClassifier(
            n_estimators=900, max_features="sqrt", n_jobs=8, random_state=42
        )
        base_model.fit(X_tr, y_tr)

        # Reporte sin calibrar
        preds_vanilla = base_model.predict(X_test_scaled)
        print("Reporte de clasificación de modelo sin calibrar:")
        print(classification_report(y_test, preds_vanilla))

        base_proba_cal = base_model.predict_proba(X_cal)
        base_proba_test = base_model.predict_proba(X_test_scaled)
        classes = base_model.classes_
        y_test_encoded = pd.Series(pd.Categorical(y_test, categories=classes).codes)

        # Métricas helper (multiclase)
        def multiclass_logloss(y_true, proba, labels):
            y_idx = pd.Series(pd.Categorical(y_true, categories=labels).codes).to_numpy()
            # evita valores 0/1 extremos
            proba = np.clip(proba, 1e-12, 1 - 1e-12)
            return log_loss(y_idx, proba, labels=np.arange(len(labels)))

        def multiclass_brier(y_true, proba, labels):
            y_idx = pd.Series(pd.Categorical(y_true, categories=labels).codes).to_numpy()
            y_bin = label_binarize(y_idx, classes=np.arange(len(labels)))
            return np.mean(np.sum((proba - y_bin) ** 2, axis=1))

        # ── Probar calibradores ────────────────────────────────────────────────────
        results = []
        best = {"name": None, "metrics": None, "predict_proba": None, "wrapper": None}

        # 1) Sigmoid / Isotonic con CalibratedClassifierCV (prefit)
        for m in ("sigmoid", "isotonic"):
            if m not in methods:
                continue
            clf = CalibratedClassifierCV(base_model, method=m, cv="prefit")
            clf.fit(X_cal, y_cal)
            p_test = clf.predict_proba(X_test_scaled)
            row = {
                "method": m,
                "log_loss": multiclass_logloss(y_test, p_test, classes),
                "brier": multiclass_brier(y_test, p_test, classes),
                "ece": expected_calibration_error(y_test, p_test, n_bins=15),
            }
            results.append(row)
            if (best["metrics"] is None) or (row["brier"] < best["metrics"]["brier"]):
                best = {"name": m, "metrics": row, "predict_proba": clf.predict_proba, "wrapper": clf}

        # 2) Beta Calibration (si está disponible)
        if "beta" in methods:
            beta_fn = fit_beta_calibration(y_cal.values, base_proba_cal)
            if beta_fn is not None:
                p_test = beta_fn(base_proba_test)
                row = {
                    "method": "beta",
                    "log_loss": multiclass_logloss(y_test, p_test, classes),
                    "brier": multiclass_brier(y_test, p_test, classes),
                    "ece": expected_calibration_error(y_test, p_test, n_bins=15),
                }
                results.append(row)
                if row["brier"] < best["metrics"]["brier"]:
                    # wrap en objeto con interfaz mínima (solo predict_proba)
                    class _BetaWrapper:
                        def __init__(self, fn): self._fn = fn

                        def predict_proba(self, X): return fn(base_model.predict_proba(X))

                    fn = beta_fn
                    beta_wrapper = _BetaWrapper(beta_fn)
                    best = {"name": "beta", "metrics": row,
                            "predict_proba": lambda X: beta_fn(base_model.predict_proba(X)),
                            "wrapper": beta_wrapper}
            else:
                print("Aviso: 'betacal' no está instalado; se omite Beta Calibration.")

        # 3) Histogram/Quantile binning con suavizado
        if "histbin" in methods:
            hist = HistogramBinningCalibrator(n_bins=20, alpha=0.15, quantile=True)
            hist.fit(pd.Series(pd.Categorical(y_cal, categories=classes).codes).values, base_proba_cal)
            p_test = hist.predict_proba(base_proba_test)
            row = {
                "method": "histbin_q20_a0.15",
                "log_loss": multiclass_logloss(y_test, p_test, classes),
                "brier": multiclass_brier(y_test, p_test, classes),
                "ece": expected_calibration_error(y_test, p_test, n_bins=15),
            }
            results.append(row)
            if row["brier"] < best["metrics"]["brier"]:
                class _HistWrapper:
                    def __init__(self, h): self.h = h

                    def predict_proba(self, X): return self.h.predict_proba(base_model.predict_proba(X))

                best = {"name": "histbin_q20_a0.15", "metrics": row,
                        "predict_proba": lambda X: hist.predict_proba(base_model.predict_proba(X)),
                        "wrapper": _HistWrapper(hist)}

        # ── Mostrar tabla de resultados ────────────────────────────────────────────
        print("\n=== Comparación de calibradores (menor es mejor) ===")
        df_res = pd.DataFrame(results).sort_values("brier")
        print(df_res.to_string(index=False))

        # ── Reporte clasificador calibrado elegido ────────────────────────────────
        # Usa el mejor calibrador para predicción de clases (argmax de proba)
        p_best = best["predict_proba"](X_test_scaled)
        y_pred_best = classes[np.argmax(p_best, axis=1)]
        print(f"\nMejor calibrador: {best['name']}")
        print("Métricas del mejor calibrador:", best["metrics"])
        print("\nReporte de clasificación con calibración elegida:")
        print(classification_report(y_test, y_pred_best, labels=classes))

        # Guarda el calibrador elegido
        # Si es CalibratedClassifierCV, lo guardamos directamente; si es beta/hist, creamos un wrapper sencillo.
        self.model = best["wrapper"] if best["wrapper"] is not None else CalibratedClassifierCV(base_model,
                                                                                                method=best["name"],
                                                                                                cv="prefit").fit(X_cal,
                                                                                                                 y_cal)

        # ── Curvas de calibración (opcional) ──────────────────────────────────────
        if display_graphic:
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve

            proba_base = base_proba_test
            proba_best = p_best
            clases = classes

            plt.figure(figsize=(7, 5))
            for target_class in clases:
                idx = list(clases).index(target_class)
                y_true_bin = np.array([1 if yt == target_class else 0 for yt in y_test])

                # Sin calibrar
                prob_pred_base, prob_true_base = calibration_curve(
                    y_true_bin, proba_base[:, idx], n_bins=10, strategy='uniform'
                )
                plt.plot(prob_pred_base, prob_true_base, marker='o', label=f"Sin calibrar '{target_class}'")

                # Calibrado (mejor)
                prob_pred_cal, prob_true_cal = calibration_curve(
                    y_true_bin, proba_best[:, idx], n_bins=10, strategy='uniform'
                )
                plt.plot(prob_pred_cal, prob_true_cal, marker='o', linestyle='--', label=f"Calibrado '{target_class}'")

            fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.plot([0, 1], [0, 1], 'k--', label='Perfecto')
            plt.xlabel('Probabilidad predicha')
            plt.ylabel('Frecuencia empírica')
            plt.title(f'Curvas de calibración (base vs mejor)\nEntrenado: {fecha_actual}')
            plt.legend()
            plt.grid()
            plt.show()

        return self.model

    def visualize_results(self, top_n=20):
        """
        Visualiza los resultados más importantes
        """
        if not self.results:
            print("No hay resultados para visualizar")
            return

        df_results = pd.DataFrame(self.results)

        # Top combinaciones por confianza
        top_results = df_results.nlargest(top_n, 'confidence')

        # Crear visualizaciones
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Combinaciones por Confianza',
                            'Distribución de Tendencias',
                            'Confianza por Tendencia Dominante',
                            'Samples vs Confianza'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )

        # Gráfico 1: Top combinaciones
        fig.add_trace(
            go.Bar(x=list(range(len(top_results))),
                   y=top_results['confidence'],
                   text=top_results['dominant_trend'],
                   name='Confianza'),
            row=1, col=1
        )

        # Gráfico 2: Distribución de tendencias
        trend_counts = df_results['dominant_trend'].value_counts()
        fig.add_trace(
            go.Pie(labels=trend_counts.index,
                   values=trend_counts.values,
                   name='Tendencias'),
            row=1, col=2
        )

        # Gráfico 3: Box plot de confianza por tendencia
        for trend in df_results['dominant_trend'].unique():
            trend_data = df_results[df_results['dominant_trend'] == trend]
            fig.add_trace(
                go.Box(y=trend_data['confidence'],
                       name=trend),
                row=2, col=1
            )

        # Gráfico 4: Samples vs Confianza
        fig.add_trace(
            go.Scatter(x=df_results['total_samples'],
                       y=df_results['confidence'],
                       mode='markers',
                       text=df_results['dominant_trend'],
                       name='Combinaciones'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True,
                          title_text="Análisis de Patrones Estocásticos")
        fig.show()

    def evaluate_positional_relation(self, df, relation_type, k_col, d_col, e_col):
        """
        Evalúa relaciones posicionales entre indicadores
        """
        tolerance = 3.0  # Tolerancia para considerar valores "iguales"

        if relation_type == 'k>d>e':
            return (df[k_col] > df[d_col]) & (df[d_col] > df[e_col])
        elif relation_type == 'e>d>k':
            return (df[e_col] > df[d_col]) & (df[d_col] > df[k_col])
        elif relation_type == 'any':
            return pd.Series([True] * len(df), index=df.index)
        else:
            return pd.Series([False] * len(df), index=df.index)

    def evaluate_timeframe_relation(self, df, relation_type):
        """
        Evalúa relaciones entre diferentes marcos temporales
        """
        if relation_type == 'both_bullish':
            return ((df['k'] > df['d']) & (df['d'] > df['e'])) & \
                ((df['k100'] > df['d100']) & (df['d100'] > df['e100']))
        elif relation_type == 'both_bearish':
            return ((df['e'] > df['d']) & (df['d'] > df['k'])) & \
                ((df['e100'] > df['d100']) & (df['d100'] > df['k100']))
        elif relation_type == 'stoch14_bull_100_bear':
            return ((df['k'] > df['d']) & (df['d'] > df['e'])) & \
                ((df['e100'] > df['d100']) & (df['d100'] > df['k100']))
        elif relation_type == 'stoch14_bear_100_bull':
            return ((df['e'] > df['d']) & (df['d'] > df['k'])) & \
                ((df['k100'] > df['d100']) & (df['d100'] > df['e100']))
        elif relation_type == 'convergence':
            # Ambos marcos temporales moviéndose en la misma dirección
            stoch14_bullish = (df['k'] > df['d']) & (df['d'] > df['e'])
            stoch100_bullish = (df['k100'] > df['d100']) & (df['d100'] > df['e100'])
            return (stoch14_bullish & stoch100_bullish) | \
                ((~stoch14_bullish) & (~stoch100_bullish))
        elif relation_type == 'divergence':
            stoch14_bullish = (df['k'] > df['d']) & (df['d'] > df['e'])
            stoch100_bullish = (df['k100'] > df['d100']) & (df['d100'] > df['e100'])
            return (stoch14_bullish & (~stoch100_bullish)) | \
                ((~stoch14_bullish) & stoch100_bullish)
        elif relation_type == 'any':
            return pd.Series([True] * len(df), index=df.index)
        else:
            return pd.Series([False] * len(df), index=df.index)

    def analyze_specific_patterns(self, df):
        """
        Analiza patrones específicos como el ejemplo mencionado
        """
        patterns = []

        # Patrón bajista específico: K14 en zona baja < D14 en zona baja < E14 en zona alta + E100 descendiendo
        bearish_pattern_1 = (
                (df['k'] >= 0) & (df['k'] < 20) &  # K14 en zona sobreventa
                (df['d'] >= 0) & (df['d'] < 20) &  # D14 en zona sobreventa
                (df['e'] >= 60) & (df['e'] < 80) &  # E14 en zona alta
                (df['k'] < df['d']) & (df['d'] < df['e']) &  # K14 < D14 < E14
                (df['e100_direction'] == 'descending') &  # E100 descendiendo
                df['future_trend'].notna()
        )

        # Patrón alcista específico: K14 en zona alta > D14 en zona alta > E14 en zona baja + E100 ascendiendo
        bullish_pattern_1 = (
                (df['k'] >= 80) & (df['k'] <= 100) &  # K14 en zona sobrecompra
                (df['d'] >= 80) & (df['d'] <= 100) &  # D14 en zona sobrecompra
                (df['e'] >= 20) & (df['e'] < 40) &  # E14 en zona baja
                (df['k'] > df['d']) & (df['d'] > df['e']) &  # K14 > D14 > E14
                (df['e100_direction'] == 'ascending') &  # E100 ascendiendo
                df['future_trend'].notna()
        )

        # Analizar patrones
        for pattern_name, pattern_mask in [('bearish_specific', bearish_pattern_1),
                                           ('bullish_specific', bullish_pattern_1)]:
            filtered_data = df[pattern_mask]

            if len(filtered_data) >= 10:
                trend_distribution = filtered_data['future_trend'].value_counts()
                total_samples = len(filtered_data)

                pattern_result = {
                    'pattern_name': pattern_name,
                    'total_samples': total_samples,
                    'trend_distribution': trend_distribution.to_dict(),
                    'accuracy': (trend_distribution.get(pattern_name.split('_')[0], 0) / total_samples) * 100
                }
                patterns.append(pattern_result)

        return patterns

    def get_current_recommendation(self, current_k14, current_d14, current_e14,
                                   current_k100, current_d100, current_e100,
                                   current_e100_direction, current_e14_direction):
        """
        Obtiene recomendación basada en valores actuales de los indicadores
        """
        if not self.results:
            return "No hay resultados de análisis disponibles"

        # Buscar combinaciones que coincidan con los valores actuales
        matching_combinations = []

        for result in self.results:
            if result.get('type') == 'zone_positional':
                # Verificar si los valores actuales coinciden con esta combinación
                k14_match = (result['k14_min'] <= current_k14 < result['k14_max'])
                d14_match = (result['d14_min'] <= current_d14 < result['d14_max'])
                e14_match = (result['e14_min'] <= current_e14 < result['e14_max'])
                k100_match = (result['k100_min'] <= current_k100 < result['k100_max'])
                d100_match = (result['d100_min'] <= current_d100 < result['d100_max'])
                e100_match = (result['e100_min'] <= current_e100 < result['e100_max'])

                direction_e100_match = (result['e100_direction'] == current_e100_direction)
                direction_e14_match = (result['e14_direction'] == current_e14_direction)  # Agregar esta línea

                # Verificar relaciones posicionales
                stoch14_relation_match = self.check_relation_match(
                    current_k14, current_d14, current_e14, result['stoch14_relation']
                )
                stoch100_relation_match = self.check_relation_match(
                    current_k100, current_d100, current_e100, result['stoch100_relation']
                )

                if (k14_match and d14_match and e14_match and k100_match and
                        d100_match and e100_match and direction_e100_match and direction_e14_match and  # Modificar esta línea
                        stoch14_relation_match and stoch100_relation_match):
                    matching_combinations.append(result)

        if not matching_combinations:
            return "No se encontraron patrones coincidentes"

        # Ordenar por confianza y obtener la mejor recomendación
        best_match = max(matching_combinations, key=lambda x: x['confidence'])

        recommendation = {
            'dominant_trend': best_match['dominant_trend'],
            'confidence': best_match['confidence'],
            'total_samples': best_match['total_samples'],
            'distribution': {
                'bullish': best_match['bullish_pct'],
                'bearish': best_match['bearish_pct'],
                'lateral': best_match['lateral_pct']
            }
        }

        return recommendation

    def check_relation_match(self, k_val, d_val, e_val, relation_type):
        """
        Verifica si los valores actuales coinciden con la relación especificada
        """
        tolerance = 3.0

        if relation_type == 'k>d>e':
            return k_val > d_val > e_val
        elif relation_type == 'e>d>k':
            return e_val > d_val > k_val
        elif relation_type == 'any':
            return True
        else:
            return False

    def predict_realtime(self,
                         symbol: str = "AVAXUSDT",
                         interval: str = "5",
                         lookback_bars: int = 200,
                         thresholds: dict = None,
                         default_class: str = "lateral"):
        """
        Actualiza el histórico y predice la última vela usando umbrales por clase.
        Asegura que la última fila NO se use para el entrenamiento (previene data leakage).
        """
        if lookback_bars < 100:
            print(
                f"WARNING: lookback_bars={lookback_bars} es menor que 100.\n"
                "No se podrá calcular correctamente el estocástico 100 y los indicadores largos pueden fallar.\n"
                "Recomiendo usar al menos 110 o 120 para evitar NaN."
            )

        # Umbrales por defecto si no se pasan
        if thresholds is None:
            thresholds = {"bearish": 0.55, "bullish": 0.55, "lateral": 0.40}

        # 1) Descargar datos
        df_new = self.get_bybit_data_private(symbol, interval, start_date=None, end_date=None, limit=lookback_bars)
        if df_new is None or len(df_new) == 0:
            raise RuntimeError("No se pudieron obtener velas de Bybit")

        # 2) Indicadores y zonas
        df_new = self.add_indicators(df_new)
        df_new = self.add_volatility_zones(df_new)

        # 3) Preparar features
        df_ready, features = self.preparar_features(df_new)
        if df_ready.empty:
            raise RuntimeError("No hay suficientes datos para calcular todas las features necesarias (lags, etc).")

        # 4) Última fila válida
        last_row = df_ready[features].iloc[-1:]
        last_meta = df_new.iloc[-1]

        # 5) Escalar y obtener probabilidades
        last_scaled = self.scaler.transform(last_row)
        proba = self.model.predict_proba(last_scaled)[0]
        classes = list(self.model.classes_)
        probabilities = {cls: float(round(p, 4)) for cls, p in zip(classes, proba)}

        # 6) Predicción con umbrales adaptativos
        thr = thresholds or getattr(self, "thresholds_", {'bearish': 0.50, 'bullish': 0.50, 'lateral': 0.38})
        pred_class = self._apply_thresholds(
            proba, list(self.model.classes_), thr,
            default_class="lateral", global_floor=0.51, min_margin=0.03
        )

        # 7) Filtro de “fake” señales con EMA_10 (mantengo tu lógica original)
        close = float(last_meta['close'])
        ema_10 = float(last_meta['EMA_10'])

        pred = pred_class
        if pred_class == 'bullish' and close < ema_10 and probabilities.get('bullish', 0) < self.umbral_strong:
            print(f"Fake bullish (cierre < EMA10 y prob<0.84): {probabilities.get('bullish')}")
            pred = 'lateral'
        elif pred_class == 'bearish' and close > ema_10 and probabilities.get('bearish', 0) < self.umbral_strong:
            print(f"Fake bearish (cierre > EMA10 y prob<0.84): {probabilities.get('bearish')}")
            pred = 'lateral'

        result = {
            "prediction": pred,
            "probabilities": probabilities,
            "thresholds": thresholds,
            "timestamp": last_meta['datetime'],
            "last_row_atr": float(last_meta['ATR']),
            "last_row_close": float(last_meta['close']),
            "raw_df": df_new[["datetime","open","high","low","close","volume"]].copy()
        }
        return result

    def _pnl_usdt_linear(self, side, entry, exit_, qty, fee_rate=0.0006, include_both_sides=True):
        """
        PnL para perp lineal USDT.
        side: 'long'/'short'
        fee_rate: comisión por lado (p.ej., 0.0006 = 6 bps).
        include_both_sides=True → descuenta entrada y salida (2 * fee_rate).
        """
        if qty is None:
            return None
        sign = 1 if side == "long" else -1
        gross = (exit_ - entry) * qty * sign
        fees = 0.0
        if include_both_sides:
            notional_entry = entry * qty
            notional_exit = exit_ * qty
            fees = fee_rate * (notional_entry + notional_exit)
        else:
            # solo fee de cierre (por si el broker ya te descontó la de entrada)
            notional_exit = exit_ * qty
            fees = fee_rate * notional_exit
        return gross - fees

    def _resolve_tp_sl_order_in_bar(self, symbol, interval_start_dt, interval_end_dt, side, tp, sl):
        """
        Baja a 1m entre interval_start_dt y interval_end_dt para resolver si tocó TP o SL primero.
        Devuelve 'tp' | 'sl' | None (si no se tocó nada).
        """
        try:
            # Bybit espera timestamps en ms; convertimos:
            start_date = interval_start_dt.strftime("%Y-%m-%d")
            end_date = interval_end_dt.strftime("%Y-%m-%d")
            # mejor usar modo rango con 1m y recortar por timestamps después
            df_1m = self.get_bybit_data_private(symbol=symbol, interval="1",
                                                start_date=start_date, end_date=end_date, limit=0)
            if df_1m is None or df_1m.empty:
                return None
            # recorta exactamente el rango
            mask = (df_1m["datetime"] >= interval_start_dt) & (df_1m["datetime"] <= interval_end_dt)
            df_1m = df_1m.loc[mask].sort_values("datetime")
            for _, r in df_1m.iterrows():
                high = float(r["high"]);
                low = float(r["low"])
                # orden por prioridad dentro de 1m: si se tocan ambos dentro del MINUTO,
                # asumimos que primero tocó el nivel más cercano al precio de apertura de ese 1m
                # (si quieres ultra-precisión necesitarías trades-ticks)
                open_ = float(r["open"])
                # chequeo “ambos tocados”
                hit_tp = (high >= tp) if side == "long" else (low <= tp)
                hit_sl = (low <= sl) if side == "long" else (high >= sl)
                if hit_tp and not hit_sl:
                    return "tp"
                if hit_sl and not hit_tp:
                    return "sl"
                if hit_tp and hit_sl:
                    # decide por cercanía al open
                    dist_tp = abs(tp - open_)
                    dist_sl = abs(sl - open_)
                    return "tp" if dist_tp < dist_sl else "sl"
            return None
        except Exception:
            return None

    def check_and_close_from_market(self, symbol: str, df_last: pd.DataFrame):
        """
        Usa la última/s velas (5m) ya descargadas para verificar si el TP/SL de la posición abierta se tocó.
        Si cierra, calcula PnL y llama tradelogger.log_close(...).
        """
        pos = self.tradelogger.get_open_position(symbol)
        if not pos:
            return  # no hay posición

        side = pos.get("side")
        entry_price = float(pos.get("entry_price", 0))
        tp = float(pos.get("take_profit", 0))
        sl = float(pos.get("stop_loss", 0))
        qty = pos.get("qty")
        fee_rate = pos.get("fee_rate", getattr(self, "fee_rate", 0.0006))
        trade_id = pos.get("trade_id")
        if not side or not entry_price or not qty or not tp or not sl:
            return  # info insuficiente

        # Asegúrate de tener al menos 1 vela cerrada reciente:
        if df_last is None or df_last.empty:
            return

        # Tomamos la última vela cerrada y, opcionalmente, también la anterior por si la posición se abrió “entre velas”
        # (si quieres, puedes filtrar desde la hora de apertura de la posición)
        df = df_last.sort_values("datetime").tail(2)

        # Recorremos las velas en orden cronológico
        for i in range(len(df)):
            row = df.iloc[i]
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_open_dt = pd.to_datetime(row["datetime"])
            # fin del intervalo = open + 5 minutos
            bar_end_dt = bar_open_dt + pd.Timedelta(
                minutes=int(df["datetime"].diff().median() / pd.Timedelta(minutes=1) if len(df) > 1 else 5))

            # ¿tocó niveles en esta vela?
            hit_tp = (bar_high >= tp) if side == "long" else (bar_low <= tp)
            hit_sl = (bar_low <= sl) if side == "long" else (bar_high >= sl)

            if not hit_tp and not hit_sl:
                continue  # nada en esta barra

            # Si ambos en la misma vela, resuelve bajando a 1m
            if hit_tp and hit_sl:
                first = self._resolve_tp_sl_order_in_bar(symbol, bar_open_dt, bar_end_dt, side, tp, sl)
                if first is None:
                    # regla conservadora: prioriza SL (peor caso)
                    first = "sl"
            else:
                first = "tp" if hit_tp else "sl"

            # Determina precio de salida (si quieres precio exacto del nivel, usa tp/sl; si prefieres close real, usa bar_close)
            exit_reason = first
            exit_price = tp if first == "tp" else sl

            # Calcula PnL
            pnl_usdt = self._pnl_usdt_linear(side, entry_price, exit_price, qty, fee_rate=fee_rate,
                                             include_both_sides=True)

            # (Opcional) PnL en R si dejaste reward_risk en la señal:
            pnl_r = None
            # Si guardaste el R “unitario” (= distancia a SL), puedes hacer:
            # R = abs(entry - sl); pnl_r = (exit - entry)/R * sign
            try:
                R = abs(entry_price - sl) if side == "long" else abs(sl - entry_price)
                sign = 1 if side == "long" else -1
                pnl_r = ((exit_price - entry_price) * sign) / (R if R != 0 else 1e-6)
            except Exception:
                pass

            # Loguea cierre y limpia posición
            self.tradelogger.log_close(
                symbol=symbol,
                exit_reason=exit_reason,
                exit_price=exit_price,
                pnl_usdt=float(pnl_usdt) if pnl_usdt is not None else None,
                pnl_r=float(pnl_r) if pnl_r is not None else None,
                order_id=pos.get("order_id"),
                trade_id=trade_id
            )
            # Si además cierras en el broker vía API, puedes dispararlo aquí.
            break  # ya cerramos; sal del loop

    def send_email(self, subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = email_sender
        msg['To'] = EMAIL_RECEIVER
        try:
            with smtplib.SMTP(EMAIL_SMTP, EMAIL_PORT) as server:
                server.starttls()
                server.login(user=email_sender, password=email_password)
                server.sendmail(email_sender, EMAIL_RECEIVER, msg.as_string())
            print("Alerta email por enviada ✅")
        except Exception as e:
            print(f"Error enviando email: {e}")

    def monitor_predictions(self, symbol="AVAXUSDT", interval="5"):
        last_prediction = None
        last_minute_checked = None
        min_margin = getattr(self, "min_margin", 0.05)

        while True:
            now = datetime.now()
            minute = now.minute
            second = now.second

            if (minute % 5 == 4) and (last_minute_checked != minute):
                # Espera cierre de vela
                if second > 0:
                    time.sleep(60 - second)

                # === Recalcula 'now' tras dormir ===
                now = datetime.now()

                # 1) Predicción en vivo
                try:
                    resultado = self.predict_realtime(symbol=symbol, interval=interval)
                    self.check_and_close_from_market(symbol, resultado.get("raw_df"))
                except Exception as e:
                    self.tradelogger.log_info(event="predict_error", error=str(e))
                    time.sleep(5)
                    continue

                pred = resultado.get('prediction')
                probs = resultado.get('probabilities', {}) or {}
                proba_pred = float(probs.get(pred, 0.0))

                ordered = sorted(probs.values(), reverse=True)
                top1 = ordered[0] if ordered else 0.0
                top2 = ordered[1] if len(ordered) > 1 else 0.0
                margin = (top1 - top2)
                margin_ok = (margin >= min_margin)

                # 2) Timestamp a Bogotá
                utc_time = resultado.get('timestamp')
                ts = pd.Timestamp(utc_time)
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                bogota_time = ts.tz_convert('America/Bogota')

                # 3) Cálculos de RR/Kelly (con guardas de ATR)
                atr = resultado.get('last_row_atr')
                close_price = resultado.get('last_row_close')
                atr_ok = (atr is not None) and (atr > 0)

                # Evita asignar 'short' por defecto cuando es lateral
                side = "long" if pred == "bullish" else ("short" if pred == "bearish" else None)
                margin_pp = margin * 100.0

                # Calcula TP/SL solo si es operable
                take_profit = stop_loss = reward_risk = kelly_fraction = None
                eligible = (side is not None) and margin_ok and atr_ok

                if eligible:
                    profit_factor = self.atr_mult
                    loss_factor = self.atr_stop
                    if side == "long":
                        take_profit = close_price + (profit_factor * atr)
                        stop_loss = close_price - (loss_factor * atr)
                    else:
                        take_profit = close_price - (profit_factor * atr)
                        stop_loss = close_price + (loss_factor * atr)

                    ganancia = abs(take_profit - close_price)
                    perdida = abs(stop_loss - close_price)
                    reward_risk = ganancia / (perdida if perdida != 0 else 1e-6)
                    p = proba_pred
                    q = 1 - p
                    kelly_fraction = max(0.0, min(p - (q / reward_risk), 0.95))

                # Gate sólo si es elegible
                gate = None
                if eligible:
                    gate = "strong" if proba_pred >= self.umbral_strong else (
                        "soft" if proba_pred >= self.umbral_soft else None)

                # 1) Log de señal (marca elegibilidad)
                trade_id = self.tradelogger.log_signal(
                    symbol=symbol, timeframe=interval, side=side, prob=proba_pred,
                    margin_pp=margin_pp, probs_dict=probs, kelly=kelly_fraction, rr=reward_risk,
                    take_profit=take_profit, stop_loss=stop_loss, price=close_price,
                    reason="pred_change" if pred != last_prediction else "no_change_reentry",
                    gate=gate
                )
                # También puedes registrar la elegibilidad explicitamente:
                # self.tradelogger.log_info(event="signal_meta", trade_id=trade_id, eligible=bool(eligible),
                #                           margin_ok=bool(margin_ok), atr_ok=bool(atr_ok))

                # 4) Decisión de trade
                if pred != last_prediction:
                    print("\n")
                    print(f"[{bogota_time.strftime('%Y-%m-%d %H:%M:%S')}] NUEVA PREDICCIÓN: {pred.upper()} "
                          f"(prob: {proba_pred * 100:.1f}%) | margen: {margin_pp:.1f} pp")
                    print("Detalle probabilidades:", probs)
                    if eligible and gate in ("soft", "strong"):
                        if self.tradelogger.has_open_position(symbol):
                            self.tradelogger.log_info(note="Señal ignorada: ya hay posición abierta",
                                                      symbol=symbol, side=side, trade_id=trade_id)
                        else:
                            self.start_order(pred, symbol, interval, bogota_time, proba_pred, probs,
                                             atr, close_price, True, trade_id=trade_id)
                    else:
                        print("Señal no elegible (lateral, sin margen, sin ATR o bajo umbral).")
                    last_prediction = pred

                else:
                    print(f"[{bogota_time.strftime('%Y-%m-%d %H:%M:%S')}] 'Sin cambios': {pred.upper()} "
                          f"(prob: {proba_pred * 100:.1f}%) | margen: {margin_pp:.1f} pp")
                    print(probs)

                    # Re-entrada solo si sube a strong y sigue elegible
                    if eligible and proba_pred >= self.umbral_strong:
                        if self.tradelogger.has_open_position(symbol):
                            self.tradelogger.log_info(note="Señal ignorada: ya hay posición abierta",
                                                      symbol=symbol, side=side, trade_id=trade_id)
                        else:
                            self.start_order(pred, symbol, interval, bogota_time, proba_pred, probs,
                                             atr, close_price, True, trade_id=trade_id)
                    else:
                        print(f"Señal por debajo del umbral strong o no elegible — no trade.")

                last_minute_checked = minute

            else:
                time.sleep(5)

            # === Recalcula 'now' aquí también antes de ventanas de reinicio ===
            now = datetime.now()

            # Ventanas de reinicio
            if (now.hour in {RESTART_HOUR_1 - 1, RESTART_HOUR_2 - 1, RESTART_HOUR_3 - 1, RESTART_HOUR_4 - 1} and now.minute == 56):
                break

            if now.hour == 0 and now.minute == 59:
                today_str_ = f"{now.day:02d}"
                tomorrow_ = now + timedelta(days=1)
                tomorrow_day_ = f"{tomorrow_.day:02d}"
                mont_str_ = f"{now.month:02d}"
                filename_ = f'AVAXUSDT_5m_2024-01-01_2025-{mont_str_}-{today_str_}.csv'
                if os.path.exists(filename_):
                    name, extension_ = os.path.splitext(filename_)
                    new_name_ = name[:-2] + tomorrow_day_ + extension_
                    os.rename(filename_, new_name_)
                    print("Nuevo nombre de archivo:", new_name_)

    def start_order(self, pred, symbol, interval, bogota_time, proba, resultado, atr, close_price, real_account, trade_id=None):
        if atr is not None and atr > 0:
            if (pred == "bullish") or (pred == "bearish"):
                response = {}
                profit_factor = self.atr_mult
                loss_factor = self.atr_stop
                take_profit = close_price + (profit_factor * atr) if pred == "bullish" else close_price - (
                        profit_factor * atr)
                stop_loss = close_price - (loss_factor * atr) if pred == "bullish" else close_price + (
                        loss_factor * atr)
                # --- CÁLCULO DEL FRAGMENTO DE KELLY ---
                ganancia = abs(take_profit - close_price)
                perdida = abs(stop_loss - close_price)
                reward_risk = ganancia / perdida if perdida != 0 else 0.0001  # protección para no dividir por 0
                p = proba - 0.25
                q = 1 - p
                kelly_fraction = p - (q / reward_risk)
                # Limitar valores extremos
                kelly_fraction = max(0.0, min(kelly_fraction, 0.95))

                # Construir el payload
                payload = {
                    "action": "buy" if pred == "bullish" else "sell",
                    "take_profit": f"{take_profit:.10f}",
                    "stop_loss": f"{stop_loss:.10f}",
                    "price": f"{close_price:.3f}",
                    "symbol": symbol,
                    "kelly_fraction": round(kelly_fraction, 4)
                }

                side = "long" if pred == "bullish" else "short"

                # Tamaño/qty: la calculas tú; si ya la tienes en el webhook de respuesta, igual logueamos el attempt.
                qty_preview = None  # pon aquí tu cálculo si lo tienes (o déjalo None)
                self.tradelogger.log_order_attempt(
                    trade_id=trade_id, symbol=symbol, side=side, qty=qty_preview, payload=payload
                )

                # Hacer el POST al webhook
                if real_account:
                    # Se envia a la cuenta real y demo
                    try:
                        response = requests.post(
                            "http://52.67.38.220:80/hook-demo",
                            json=payload,
                            timeout=10
                        )
                        print("POST al webhook enviado:", payload)
                        print("Respuesta:", response.text)
                    except Exception as e:
                        print("Error enviando al webhook:", e)

                    response_data = response.json()
                    success = response_data.get('success')

                    if success:
                        # Extrae orderId si viene en la respuesta OK
                        bybit_resp = response_data.get('bybit_response', {})
                        order_id = None
                        try:
                            order_id = bybit_resp.get("result", {}).get("orderId")
                        except Exception:
                            pass

                        # Si no te llega qty exacto, puedes usar el 'qty' de la respuesta si existe:
                        qty_final = response_data.get("qty")

                        # ACK + marcar posición abierta
                        self.tradelogger.log_order_ack(
                            trade_id=trade_id, symbol=symbol, side=side, qty=qty_final, broker_raw=response_data,
                            order_id=order_id
                        )
                        # justo después de log_order_ack(...) y de conocer qty_final:
                        self.tradelogger.enrich_open_position(
                            symbol,
                            entry_price=close_price,
                            take_profit=take_profit,
                            stop_loss=stop_loss,
                            side=side,
                            fee_rate=getattr(self, "fee_rate", 0.0006),  # 6 bps por lado si no defines otro
                        )
                        subject = f"ALERTA TRADING DEMO CON REGISTRO: Tendencia {pred.upper()} >{self.umbral_soft * 100:.2f}%"
                        body = f"""
                        Predicción automática:
                        Vela: {bogota_time.strftime('%Y-%m-%d %H:%M:%S')}
                        Tendencia: {pred.upper()}
                        Probabilidad: {proba * 100:.2f}%
                        Reward/Risk: {reward_risk:.2f}
                        Fragmento Kelly: {kelly_fraction:.4f}
                        Detalle probabilidades: {resultado}
                        Json: {payload}
                        """
                        self.send_email(subject, body)
                    else:
                        # Rechazo (incluye "Cambio de posición no permitido")
                        self.tradelogger.log_order_reject(
                            trade_id=trade_id, symbol=symbol, side=side, qty=response_data.get("qty"),
                            broker_raw=response_data
                        )

                else:
                    # Solo a cuenta demo
                    try:
                        response = requests.post(
                            "http://52.67.38.220:80/hook-demo",
                            json=payload,
                            timeout=10
                        )
                        print("POST al webhook enviado:", payload)
                        print("Respuesta:", response.text)
                    except Exception as e:
                        print("Error enviando al webhook:", e)

                    response_data = response.json()
                    success = response_data.get('success')

                    if success:
                        subject = f"ALERTA TRADING DEMO: Tendencia {pred.upper()} >50%"
                        body = f"""
                        Predicción automática:
                        Vela: {bogota_time.strftime('%Y-%m-%d %H:%M:%S')}
                        Tendencia: {pred.upper()}
                        Probabilidad: {proba * 100:.2f}%
                        Reward/Risk: {reward_risk:.2f}
                        Fragmento Kelly: {kelly_fraction:.4f}
                        Detalle probabilidades: {resultado}
                        Json: {payload}
                        """
                        self.send_email(subject, body)

    def append_new_data(self, new_df):
        """
        Anexa nuevos datos OHLCV al histórico acumulado (self.data), evitando duplicados.
        """
        if new_df is None or new_df.empty:
            return
        if self.data is None or self.data.empty:
            self.data = new_df
        else:
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            self.data = self.data.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    def simular_estrategia(self, start_date="2024-07-01", end_date="2025-07-20",
                           capital_inicial=1000, fee=0.0012, leverage=19,
                           verbose=True, graficar=True):
        """
        Simula la estrategia entre dos fechas con capital inicial, Kelly y apalancamiento.
        Entrena el modelo y scaler una sola vez con la data previa a start_date,
        y predice la probabilidad para cada trade con ese modelo.
        SOLO UN TRADE ABIERTO A LA VEZ.
        Exporta resultados y resumen a CSV.
        """
        df = self.data.copy()
        features = [
            'k100', 'd100', 'e100', 'k', 'd', 'e', 'volatility', 'ATR',
            'RSI_14', 'ADX_14', 'pct_green_last_n', 'pct_red_last_n',
            'avg_volume_last_n', 'pct_volume_change'
        ]

        # 1. Entrenar modelo y scaler solo con datos ANTES de start_date
        train_df = df[df['datetime'] < start_date].dropna(subset=features + ['future_trend'])
        if train_df.empty:
            raise Exception("No hay suficiente data previa al periodo de simulación para entrenar el modelo.")

        X_train = train_df[features]
        y_train = train_df['future_trend']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        model.fit(X_train_scaled, y_train)

        # 2. Filtrar solo el periodo a simular
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].reset_index(drop=True)

        capital = capital_inicial
        equity_curve = [capital]
        max_equity = capital
        trades = []

        i = 0
        while i < len(df):
            row = df.iloc[i]
            if row['future_trend'] not in ('bullish', 'bearish'):
                i += 1
                continue
            price = row['close']
            atr = row['ATR']
            if pd.isna(atr) or atr <= 0:
                i += 1
                continue

            # Calcular probabilidad real del modelo en ese momento usando SOLO datos previos
            X_test = row[features].values.reshape(1, -1)
            X_test_scaled = scaler.transform(X_test)
            probs = model.predict_proba(X_test_scaled)[0]
            clases = model.classes_

            pred_class = model.predict(X_test_scaled)[0]  # puede ser 'bullish', 'bearish', 'lateral'
            pred_probas = dict(zip(clases, probs))
            prob = pred_probas.get(pred_class, 0.5)

            if pred_class not in ('bullish', 'bearish'):
                i += 1
                continue

            # Kelly
            reward = 3 * atr
            risk = 2.8 * atr
            rrr = reward / risk if risk > 0 else 1
            q = 1 - prob
            kelly = prob - (q / rrr)
            kelly = max(0, min(kelly, 0.95))
            if kelly < 0.01:
                i += 1
                continue

            pos_size = capital * kelly * leverage
            commission = abs(pos_size * fee)

            # TP/SL
            if pred_class == 'bullish':
                tp = price + reward
                sl = price - risk
            else:
                tp = price - reward
                sl = price + risk

            # Buscar hacia adelante hasta que se alcance TP o SL (sin límite)
            df_future = df.iloc[i + 1:]
            hit_tp = hit_sl = None
            idx_tp = idx_sl = None
            salida_idx = None
            for j, fut in df_future.iterrows():
                if pred_class == "bullish":
                    if fut['high'] >= tp:
                        hit_tp = fut['datetime']
                        idx_tp = j
                        salida_idx = i + 1 + j
                        break
                    if fut['low'] <= sl:
                        hit_sl = fut['datetime']
                        idx_sl = j
                        salida_idx = i + 1 + j
                        break
                else:
                    if fut['low'] <= tp:
                        hit_tp = fut['datetime']
                        idx_tp = j
                        salida_idx = i + 1 + j
                        break
                    if fut['high'] >= sl:
                        hit_sl = fut['datetime']
                        idx_sl = j
                        salida_idx = i + 1 + j
                        break

            # Resultado y cierre
            if (idx_tp is not None) and (idx_sl is not None):
                if idx_tp < idx_sl:
                    pnl = pos_size * (reward / price) - commission
                    resultado = "TP"
                    salida_fecha = hit_tp
                    salida_idx = i + 1 + idx_tp
                else:
                    pnl = -pos_size * (risk / price) - commission
                    resultado = "SL"
                    salida_fecha = hit_sl
                    salida_idx = i + 1 + idx_sl
            elif idx_tp is not None:
                pnl = pos_size * (reward / price) - commission
                resultado = "TP"
                salida_fecha = hit_tp
                salida_idx = i + 1 + idx_tp
            elif idx_sl is not None:
                pnl = -pos_size * (risk / price) - commission
                resultado = "SL"
                salida_fecha = hit_sl
                salida_idx = i + 1 + idx_sl
            else:
                if not df_future.empty:
                    last_close = df_future['close'].iloc[-1]
                    salida_fecha = df_future['datetime'].iloc[-1]
                    salida_idx = len(df)  # no hay más filas
                else:
                    last_close = price
                    salida_fecha = row['datetime']
                    salida_idx = i + 1
                if pred_class == 'bullish':
                    pnl = pos_size * ((last_close - price) / price) - commission
                else:
                    pnl = pos_size * ((price - last_close) / price) - commission
                resultado = "No Exit"

            capital += pnl
            max_equity = max(max_equity, capital)
            drawdown = (max_equity - capital) / max_equity

            trade = {
                "fecha_entrada": row['datetime'],
                "fecha_salida": salida_fecha,
                "pred": pred_class,
                "entry": price,
                "tp": tp,
                "sl": sl,
                "probabilidad": prob,
                "kelly_fragment": kelly,
                "pos_size": pos_size,
                "comision": commission,
                "resultado": resultado,
                "pnl": pnl,
                "capital_post": capital,
                "drawdown": drawdown
            }
            trades.append(trade)
            equity_curve.append(capital)

            # Saltar a la siguiente fila DESPUÉS del cierre del trade
            if salida_idx is None:
                i += 1
            else:
                i = salida_idx + 1

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No hay trades en ese rango.")
            return

        # === ESTADÍSTICAS ===
        total_trades = len(trades_df)
        ganadores = (trades_df['pnl'] > 0).sum()
        perdedores = (trades_df['pnl'] < 0).sum()
        profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(
            trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if perdedores else np.nan
        max_drawdown = trades_df['drawdown'].max()
        final_capital = trades_df['capital_post'].iloc[-1]
        total_profit = final_capital - capital_inicial
        win_rate = ganadores / total_trades * 100
        avg_prob = trades_df['probabilidad'].mean()
        avg_kelly = trades_df['kelly_fragment'].mean()

        resumen = {
            "Capital inicial": capital_inicial,
            "Capital final": round(final_capital, 2),
            "Total profit/loss": round(total_profit, 2),
            "Trades": total_trades,
            "Ganadores": ganadores,
            "Perdedores": perdedores,
            "Win rate %": round(win_rate, 2),
            "Profit factor": round(profit_factor, 2),
            "Max drawdown %": round(100 * max_drawdown, 2),
            "Probabilidad promedio": round(avg_prob, 4),
            "Kelly promedio": round(avg_kelly, 4)
        }

        if verbose:
            print("\n--- RESUMEN SIMULACIÓN ---")
            for k, v in resumen.items():
                print(f"{k}: {v}")

            print(trades_df[
                      ['fecha_entrada', 'fecha_salida', 'pred', 'entry', 'tp', 'sl', 'probabilidad', 'kelly_fragment',
                       'pos_size', 'comision', 'resultado', 'pnl', 'capital_post', 'drawdown']].tail(10))

        # === GUARDAR CSVs ===

        bogota_tz = pytz.timezone('America/Bogota')
        for col in ['fecha_entrada', 'fecha_salida']:
            trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce', utc=True).dt.tz_convert(bogota_tz)
            trades_df[col] = trades_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        trades_filename = f"sim_trades_{start_date}_a_{end_date}.csv"
        resumen_filename = f"sim_resumen_{start_date}_a_{end_date}.csv"

        trades_df.to_csv(trades_filename, index=False)
        if verbose:
            print(f"\nArchivo de operaciones exportado a: {trades_filename}")

        resumen_df = pd.DataFrame([resumen])
        resumen_df.to_csv(resumen_filename, index=False)
        if verbose:
            print(f"Archivo de resumen exportado a: {resumen_filename}")

        # === GRÁFICOS ===
        if graficar:
            plt.figure(figsize=(14, 6))
            plt.subplot(2, 2, 1)
            plt.plot(equity_curve, label="Equity Curve")
            plt.title("Evolución de Capital")
            plt.xlabel("Trade #")
            plt.ylabel("Capital (USDT)")
            plt.grid()
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.hist(trades_df['pnl'], bins=30, color='royalblue', alpha=0.7)
            plt.title("Distribución PnL por Trade")
            plt.xlabel("Profit/Loss")
            plt.ylabel("Frecuencia")

            plt.subplot(2, 2, 3)
            plt.plot(trades_df['drawdown'], label="Drawdown")
            plt.title("Drawdown")
            plt.xlabel("Trade #")
            plt.ylabel("Drawdown (%)")
            plt.grid()
            plt.legend()

            plt.subplot(2, 2, 4)
            trades_df['resultado'].value_counts().plot(kind='bar')
            plt.title("Tipos de Salidas (TP/SL/No Exit)")

            plt.tight_layout()
            plt.show()

        return trades_df, resumen

    def append_new_data(self, new_df):
        """
        Anexa nuevos datos OHLCV al histórico acumulado (self.data), evitando duplicados.
        """
        if new_df is None or new_df.empty:
            return
        if self.data is None or self.data.empty:
            self.data = new_df
        else:
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            self.data = self.data.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    def preparar_features(self, df, lags=[1]):
        df = df.copy()

        base_features = [
            'd100', 'e100', 'e', 'volatility', 'ATR', 'ADX_14', 'avg_volume_last_n', 'pct_volume_change', 'k',
            'd', 'close', 'high', 'low', 'volume'
        ]

        no_lag_features = [
            'minute_sin', 'minute_cos', 'k100', 'dow_sin', 'dow_cos'
        ]

        features_testing = [
            'return_5',
            'volatility_10', 'volatility_50', 'atr_norm',
            'roc_10', 'roc_30', 'momentum_diff',
            'rsi_14', 'rsi_50',
            'ema_ratio'
        ]

        dummy_cols_base = ['trend_ema200', 'trend_vwap']
        dummy_expected = [
            'trend_ema200_up', 'trend_vwap_up'
        ]

        # === Convertir variables categóricas en dummies ===
        df = pd.get_dummies(df, columns=dummy_cols_base)

        # === Asegurar que todas las dummies estén presentes ===
        for col in dummy_expected:
            if col not in df.columns:
                df[col] = 0

        # === Generar lag features ===
        for col in base_features:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # === Lista final de features: actuales + lags ===
        lag_feature_names = [f'{col}_lag{lag}' for col in base_features for lag in lags]
        features_final = base_features + no_lag_features + lag_feature_names + features_testing

        # === Reemplazar infinitos y eliminar NaNs solo en las columnas de interés ===
        df[features_final] = df[features_final].replace([np.inf, -np.inf], np.nan)

        return df, features_final

    def _apply_thresholds(self,
                          proba_row,
                          classes,
                          thresholds,
                          default_class="lateral",
                          global_floor=0.50,  # confianza mínima global
                          min_margin=0.03):  # diferencia mínima top1 - top2

        # 1) Candidatas que superan su umbral por clase
        accepted = [(cls, float(p)) for cls, p in zip(classes, proba_row)
                    if float(p) >= float(thresholds.get(cls, 0.33))]

        if not accepted:
            return default_class

        # 2) Ordenar por prob
        accepted.sort(key=lambda x: x[1], reverse=True)
        top_cls, top_p = accepted[0]

        # 3) Confianza mínima global
        if global_floor is not None and top_p < global_floor:
            return default_class

        # 4) Margen vs. segunda mejor probabilidad (sobre TODAS las clases)
        sorted_all = sorted([float(p) for p in proba_row], reverse=True)
        second_p = sorted_all[1] if len(sorted_all) > 1 else 0.0
        if min_margin is not None and (top_p - second_p) < min_margin:
            return default_class

        return top_cls

    def simulate_thresholds(self, X_test_scaled, y_test, thresholds):
        """
        Evalúa la matriz de confusión y reporte de clasificación usando umbrales por clase.
        Requiere que self.model ya esté calibrado y entrenado.
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Primero entrena/calibra el modelo (self.model).")

        proba = self.model.predict_proba(X_test_scaled)
        classes = list(self.model.classes_)
        y_pred = [self._apply_thresholds(p, classes, thresholds) for p in proba]

        labels = ["bearish", "bullish", "lateral"]  # orden consistente
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print("=== Umbrales usados ===")
        print(thresholds)
        print("\nMatriz de confusión con umbrales:\n", cm)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        print("\nMatriz normalizada por fila:\n", np.round(cm_norm, 3))
        print("\nReporte de clasificación con umbrales:")
        print(classification_report(y_test, y_pred, labels=labels))

    def wilson_ci(self, k, n, z=1.96):
        if n == 0: return (np.nan, np.nan)
        p = k / n
        denom = 1 + z ** 2 / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
        return center - half, center + half

    def audit_bands(self, model, X_hold_scaled, y_hold, classes,
                    bins=np.arange(0.55, 0.91, 0.05),
                    min_margin=None, session_mask=None, session_name="all"):
        proba = model.predict_proba(X_hold_scaled)
        top = proba.max(axis=1)
        idx = proba.argmax(axis=1)
        cls_top = np.array([classes[i] for i in idx])
        is_trend = np.isin(cls_top, ["bullish", "bearish"])
        top2 = np.partition(proba, -2, axis=1)[:, -2]
        margin = top - top2

        y = np.asarray(y_hold)

        if session_mask is None:
            session_mask = np.ones_like(top, dtype=bool)

        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (top >= lo) & (top < hi) & is_trend & session_mask
            if min_margin is not None:
                m &= (margin >= min_margin)
            n = int(m.sum())
            if n == 0:
                rows.append((session_name, f"[{lo:.2f},{hi:.2f})", 0, np.nan, np.nan, np.nan))
                continue
            hits = int((y[m] == cls_top[m]).sum())
            prec = hits / n
            ci_lo, ci_hi = self.wilson_ci(hits, n)
            rows.append((session_name, f"[{lo:.2f},{hi:.2f})", n, prec, ci_lo, ci_hi))
        return pd.DataFrame(rows, columns=["session", "band", "n", "precision", "ci_lo", "ci_hi"])

    def _wilson_ci(self, k, n, z=1.96):
        if n <= 0: return (np.nan, np.nan)
        p = k / n
        denom = 1 + z ** 2 / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
        return center - half, center + half

    def select_thresholds_by_band_precision(
            self,
            X_hold,
            y_hold,
            already_scaled=True,
            features=None,
            band_step=0.05,
            u_min=0.55,
            u_max=0.95,
            min_margin=0.03,
            # --- pisos separados por sesión ---
            min_n_soft=1000,
            min_n_strong=1200,
            min_trade_rate_soft=0.10,
            min_trade_rate_strong=0.03,
            # objetivos (IC inferior de Wilson)
            soft_range=(0.83, 0.865),
            strong_range=(0.89, 0.91),
            soft_start=0.60,
            strong_start=0.80,
            min_gap=0.03,
            use_ci_lower=True,
            prefer_center_soft=True  # solo aplicamos “prefer center” al modo soft
    ):

        def _wilson_ci(k, n, z=1.96):
            if n <= 0: return (np.nan, np.nan)
            p = k / n
            denom = 1 + z ** 2 / n
            center = (p + z * z / (2 * n)) / denom
            half = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
            return center - half, center + half

        # --- coerción y modelo ---
        if not already_scaled:
            if features is None:
                features = getattr(self, "features_", None)
                if features is None:
                    raise ValueError("Debes pasar 'features' si already_scaled=False.")
            X_arr = np.asarray(X_hold[features].values if hasattr(X_hold, "__getitem__") else X_hold)
            X_arr = self.scaler.transform(X_arr)
        else:
            X_arr = np.asarray(X_hold.values) if hasattr(X_hold, "values") else np.asarray(X_hold)
        if X_arr.ndim != 2:
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            else:
                raise TypeError(f"Se esperaba array 2D, recibido ndim={X_arr.ndim}")

        mdl = getattr(self, "model", None)
        if mdl is None or not hasattr(mdl, "predict_proba"):
            raise RuntimeError("self.model no está entrenado o no soporta predict_proba().")

        proba = mdl.predict_proba(X_arr)
        classes = list(mdl.classes_)
        top = proba.max(axis=1)
        idx = proba.argmax(axis=1)
        cls_top = np.array([classes[i] for i in idx])
        is_trend = np.isin(cls_top, ["bullish", "bearish"])
        top2 = np.partition(proba, -2, axis=1)[:, -2]
        margin = top - top2
        y = np.asarray(y_hold)

        # --- construir bandas ---
        bins = np.arange(u_min, u_max + 1e-9, band_step)
        rows = []
        N = len(y)
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (top >= lo) & (top < hi) & is_trend & (margin >= float(min_margin))
            n = int(m.sum())
            if n == 0:
                rows.append((lo, hi, 0, np.nan, np.nan, np.nan, 0.0));
                continue
            hits = int((y[m] == cls_top[m]).sum())
            prec = hits / n
            ci_lo, ci_hi = _wilson_ci(hits, n)
            trade_rate = n / N
            rows.append((lo, hi, n, prec, ci_lo, ci_hi, trade_rate))
        dfb = pd.DataFrame(rows, columns=["u_lo", "u_hi", "n", "precision", "ci_lo", "ci_hi", "trade_rate"])

        key = "ci_lo" if use_ci_lower else "precision"
        reason_soft = "normal"
        reason_strong = "normal"

        # ---------- SOFT: centro del rango + pisos ----------
        lo_s, hi_s = map(float, soft_range)
        target_s = 0.5 * (lo_s + hi_s)
        cand_s = dfb[dfb["u_lo"] >= float(soft_start)].copy()
        feas_s = cand_s[(cand_s["n"] >= int(min_n_soft)) &
                        (cand_s["trade_rate"] >= float(min_trade_rate_soft)) &
                        (cand_s[key] >= lo_s)].copy()
        if not feas_s.empty:
            if prefer_center_soft:
                feas_s = feas_s.assign(dist=(feas_s[key] - target_s).abs())
                feas_s = feas_s.sort_values(by=["dist", "u_lo", "n"], ascending=[True, True, False])
            else:
                feas_s = feas_s.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
            soft_row = feas_s.iloc[0]
        else:
            # fallback soft: mejor ci_lo, luego n/trade
            fb_s = cand_s[cand_s["n"] > 0].copy()
            fb_s = fb_s.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
            soft_row = fb_s.iloc[0]
            reason_soft = "fallback"

        u_soft = float(np.round(soft_row["u_lo"], 3))

        # ---------- STRONG: volumen primero + pisos ----------
        lo_h, hi_h = map(float, strong_range)
        cand_h = dfb[dfb["u_lo"] >= float(strong_start)].copy()

        # 1) Cumple CI_min + pisos ⇒ elegir por mayor n/trade_rate (prioriza volumen)
        feas_h = cand_h[(cand_h["n"] >= int(min_n_strong)) &
                        (cand_h["trade_rate"] >= float(min_trade_rate_strong)) &
                        (cand_h[key] >= lo_h)].copy()
        if not feas_h.empty:
            feas_h = feas_h.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
            strong_row = feas_h.iloc[0]
        else:
            # 2) Relajar CI (pero mantener pisos) y elegir por volumen
            relax_h = cand_h[(cand_h["n"] >= int(min_n_strong)) &
                             (cand_h["trade_rate"] >= float(min_trade_rate_strong))].copy()
            if not relax_h.empty:
                relax_h = relax_h.sort_values(by=[key, "n", "trade_rate", "u_lo"],
                                              ascending=[False, False, False, True])
                strong_row = relax_h.iloc[0]
                reason_strong = "relaxed_quality_kept_floors"
            else:
                # 3) Último recurso: bajar u hasta encontrar la mejor banda por n/trade (pero ≥ strong_start)
                fb_h = cand_h[cand_h["n"] > 0].copy()
                fb_h = fb_h.sort_values(by=["n", "trade_rate", key, "u_lo"], ascending=[False, False, False, True])
                strong_row = fb_h.iloc[0]
                reason_strong = "fallback_volume_first"

        u_strong = float(np.round(strong_row["u_lo"], 3))

        # ---------- BUMP opcional de strong si ≤ 0.75 ----------
        # (se hace ANTES de forzar gap para que el gap considere el bump)
        if u_strong <= 0.75:
            u_bumped = float(np.round(u_strong + 0.05, 3))
            # buscar fila exacta para el bump
            m_b = np.isclose(dfb["u_lo"].values, u_bumped)
            if m_b.any():
                row_b = dfb.loc[m_b].iloc[0]
                # validar pisos + calidad mínima strong
                floors_ok = (int(row_b["n"]) >= int(min_n_strong)) and \
                            (float(row_b["trade_rate"]) >= float(min_trade_rate_strong))
                quality_ok = (float(row_b[key]) >= float(lo_h))
                if floors_ok and quality_ok:
                    u_strong = u_bumped
                    strong_row = row_b
                    reason_strong = "forced_bump_if_<=0.75"
                else:
                    reason_strong = "forced_bump_if_<=0.75_rejected_by_floors_or_quality"
            else:
                reason_strong = "forced_bump_if_<=0.75_no_band"

        # --- Enforce gap: soft < strong ---
        if u_soft >= u_strong:
            below = dfb[(dfb["u_lo"] < u_strong) &
                        (dfb["u_lo"] >= soft_start) &
                        (dfb["n"] >= int(min_n_soft)) &
                        (dfb["trade_rate"] >= float(min_trade_rate_soft))].copy()
            if not below.empty:
                below = below.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
                soft_row = below.iloc[0]
                u_soft = float(np.round(min(soft_row["u_lo"], u_strong - float(min_gap)), 3))
                reason_soft = reason_soft if reason_soft != "normal" else "adjusted_below_strong"
            else:
                u_soft = float(np.round(max(float(soft_start), u_strong - float(min_gap)), 3))
                # re-sync soft_row con la banda de u_soft (si existe)
                m = np.isclose(dfb["u_lo"].values, u_soft)
                if m.any():
                    soft_row = dfb.loc[m].iloc[0]
                reason_soft = reason_soft if reason_soft != "normal" else "forced_gap"

        # --- Guard post-selección STRONG: nunca devolver algo por debajo de pisos (tras bump/gap) ---
        if (int(strong_row["n"]) < int(min_n_strong)) or (
                float(strong_row["trade_rate"]) < float(min_trade_rate_strong)):
            cands = dfb[(dfb["u_lo"] >= float(strong_start)) &
                        (dfb["n"] >= int(min_n_strong)) &
                        (dfb["trade_rate"] >= float(min_trade_rate_strong)) &
                        (dfb[key] >= float(lo_h))].copy()
            if not cands.empty:
                cands = cands.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
                strong_row = cands.iloc[0]
                u_strong = float(np.round(strong_row["u_lo"], 3))
                reason_strong = (
                        reason_strong + "_adjusted_for_floors") if reason_strong != "normal" else "adjusted_for_floors"
            else:
                # último recurso: fija mínimo razonable manteniendo gap
                u_strong = float(np.round(max(0.75, u_soft + float(min_gap)), 3))
                # re-sync strong_row si existe banda
                m = np.isclose(dfb["u_lo"].values, u_strong)
                if m.any():
                    strong_row = dfb.loc[m].iloc[0]
                reason_strong = (reason_strong + "_forced_min_u75") if reason_strong != "normal" else "forced_min_u75"

        # --- RE-SYNC FINAL: asegura que las métricas correspondan a los u elegidos ---
        m_str = np.isclose(dfb["u_lo"].values, u_strong)
        if m_str.any():
            strong_row = dfb.loc[m_str].iloc[0]
        m_soft = np.isclose(dfb["u_lo"].values, u_soft)
        if m_soft.any():
            soft_row = dfb.loc[m_soft].iloc[0]

        # persistir
        self.umbral_soft = u_soft
        self.umbral_strong = u_strong

        self.threshold_policy_ = {
            "method": "band_precision_selector_strong_max_volume",
            "params": {
                "band_step": float(band_step), "u_min": float(u_min), "u_max": float(u_max),
                "min_margin": float(min_margin),
                "min_n_soft": int(min_n_soft), "min_n_strong": int(min_n_strong),
                "min_trade_rate_soft": float(min_trade_rate_soft),
                "min_trade_rate_strong": float(min_trade_rate_strong),
                "soft_range": tuple(map(float, soft_range)), "strong_range": tuple(map(float, strong_range)),
                "soft_start": float(soft_start), "strong_start": float(strong_start),
                "min_gap": float(min_gap), "use_ci_lower": bool(use_ci_lower)
            },
            "soft": {
                "reason": reason_soft,
                "chosen": {
                    "u": u_soft, "n": int(soft_row["n"]), "precision": float(soft_row["precision"]),
                    "ci": [float(soft_row["ci_lo"]), float(soft_row["ci_hi"])],
                    "band": [float(soft_row["u_lo"]), float(soft_row["u_hi"])],
                    "trade_rate": float(soft_row["trade_rate"])
                }
            },
            "strong": {
                "reason": reason_strong,
                "chosen": {
                    "u": u_strong, "n": int(strong_row["n"]), "precision": float(strong_row["precision"]),
                    "ci": [float(strong_row["ci_lo"]), float(strong_row["ci_hi"])],
                    "band": [float(strong_row["u_lo"]), float(strong_row["u_hi"])],
                    "trade_rate": float(strong_row["trade_rate"])
                }
            }
        }
        return u_strong, self.threshold_policy_["strong"], u_soft, self.threshold_policy_["soft"], dfb




def is_allowed_time():
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # Bloqueo desde sábado 17:00 (5pm) hasta lunes 06:00 (6am)
    if (weekday == 5 and hour >= 17) or (weekday == 6) or (weekday == 0 and hour < 4):
        return False
    return True


if __name__ == "__main__":
    now = datetime.now()

    # analyzer = StochasticAnalyzer(api_key=api_key_roge, api_secret=api_secret_roge)
    analyzer = StochasticAnalyzer(api_key=api_key, api_secret=api_secret)
    # API pública
    # analyzer = StochasticAnalyzer()

    today_str = f"{datetime.now().day:02d}"
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_day = f"{tomorrow.day:02d}"
    mont_str = f"{datetime.now().month:02d}"
    filename = f'AVAXUSDT_5m_2024-01-01_2025-{mont_str}-{today_str}.csv'
    if os.path.exists(filename):
        nombre, extension = os.path.splitext(filename)
        new_name = nombre[:-2] + tomorrow_day + extension
        os.rename(filename, new_name)
        print("Nuevo nombre de archivo:", new_name)

    end_date = f"2025-{mont_str}-{tomorrow_day}"
    filename_tomorrow = f'AVAXUSDT_5m_2024-01-01_{end_date}.csv'
    if os.path.exists(filename_tomorrow):
        print(f"iniciando script a las {now.hour}:{now.minute}:{now.second} horas")
        # print(f"end_date {end_date}")
        analyzer.run_analysis(
            symbol="AVAXUSDT",
            start_date="2024-01-01",
            end_date=end_date,
            interval="5"
        )

        # simulacion
        df = analyzer.data

        sim = TradeSimulator(
            analyzer,
            df,  # <-- dataset completo (170k). El simulador recorta internamente.
            initial_equity=2000,
            leverage=19,
            risk_fraction=0.02,
            retrain_each_bar=False,  # entrena UNA vez antes del test
            suppress_fit_stdout=False,  # silencia reporte de entrenamiento
            suppress_predict_stdout=True,  # silencia prints de predicción offline
            use_offline_predictor=True,  # no llama al API
            offline_lookback_bars=200,
            n_keep_for_fit=40000,
            close_trades_at_test_end=True
        )
        results = sim.run(test_start="2025-06-01", test_end="2025-08-15")
        print(results["final_equity"], results["n_trades"])

        #     # Guardar resultados
        # results_df = analyzer.save_results("avax_stochastic_analysis_cached.csv")

        # analyzer.get_best_calibration_model(display_graphic=True)


        # analyzer.create_predictive_model(display_graphic=True)  # entrena y guarda scaler/modelo



        # analyzer.monitor_predictions()
        # print("Cierre de primer loop en monitor_predictions()")
        #
        # while True:
        #     now = datetime.now()
        #     if (now.hour == RESTART_HOUR_1 or now.hour == RESTART_HOUR_2 or now.hour == RESTART_HOUR_3 or now.hour == RESTART_HOUR_4) and now.minute == 0:
        #         print("Reiniciando script programado a las ", now.hour, " horas")
        #         print("KPIs:", analyzer.tradelogger.compute_stats(False, True))
        #         today_str = f"{datetime.now().day:02d}"
        #         tomorrow = datetime.now() + timedelta(days=1)
        #         tomorrow_day = f"{tomorrow.day:02d}"
        #         mont_str = f"{datetime.now().month:02d}"
        #         end_date = f"2025-{mont_str}-{tomorrow_day}"
        #         print(f"tomorrow will be {end_date}")
        #         filename_tomorrow = f'AVAXUSDT_5m_2024-01-01_{end_date}.csv'
        #         if os.path.exists(filename_tomorrow):
        #             analyzer = StochasticAnalyzer(api_key=api_key, api_secret=api_secret)
        #             analyzer.run_analysis(
        #                 symbol="AVAXUSDT",
        #                 start_date="2024-01-01",
        #                 end_date=end_date,
        #                 interval="5"
        #             )
        #             analyzer.create_predictive_model()
        #             analyzer.monitor_predictions()
        #         else:
        #             print(f"El archivo {filename_tomorrow} no existe")
        #     time.sleep(60)
    else:
        print(f"El archivo {filename} no existe")
