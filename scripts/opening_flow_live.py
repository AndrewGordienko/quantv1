"""Run one Opening Flow paper decision pass (10:00 ET)."""

from quantv1.forward.opening_flow_live import close_due, mark_once, reconcile, run_loop, run_once


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--send-orders", action="store_true")
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--mark", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--notional", type=float, default=10_000.0)
    args = parser.parse_args()
    if args.loop:
        run_loop(send_orders=args.send_orders, notional=args.notional)
    elif args.close:
        close_due(send_orders=args.send_orders)
    elif args.reconcile:
        reconcile()
    elif args.mark:
        mark_once()
    else:
        run_once(send_orders=args.send_orders, notional=args.notional)
