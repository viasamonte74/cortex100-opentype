"""opentype-miner CLI."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="opentype-miner")
    sub = p.add_subparsers(dest="cmd")

    # Leaf parsers only for help routing; each command re-parses its own argv.
    sub.add_parser("recon", help="refresh RECON.md / fetch structured_server")
    d = sub.add_parser("data", help="data engine")
    d_sub = d.add_subparsers(dest="data_cmd")
    d_sub.add_parser("generate")
    d_sub.add_parser("oracle-chat")
    d_sub.add_parser("bank")
    d_sub.add_parser("hard-mine")
    e = sub.add_parser("eval", help="simulator / parity")
    e_sub = e.add_subparsers(dest="eval_cmd")
    e_sub.add_parser("sim")
    e_sub.add_parser("parity")
    t = sub.add_parser("train", help="fine-tune")
    t_sub = t.add_subparsers(dest="train_cmd")
    t_sub.add_parser("reads")
    t_sub.add_parser("harness")
    t_sub.add_parser("wise-ft")
    sub.add_parser("export", help="merge LoRA → duel-legal BF16 export")

    if not argv:
        p.print_help()
        raise SystemExit(2)

    head = argv[0]
    rest = argv[1:]
    if head == "recon":
        from opentype_miner.recon import main as recon_main

        recon_main(rest)
        return
    if head == "data":
        if not rest:
            raise SystemExit("data subcommands: generate | oracle-chat | bank | hard-mine")
        which, tail = rest[0], rest[1:]
        if which == "generate":
            from opentype_miner.data.generate import main as m
        elif which == "oracle-chat":
            from opentype_miner.data.oracle_chat import main as m
        elif which == "bank":
            from opentype_miner.data.bank_ingest import main as m
        elif which == "hard-mine":
            from opentype_miner.data.hard_mine import main as m
        else:
            raise SystemExit(f"unknown data command {which}")
        m(tail)
        return
    if head == "eval":
        if not rest:
            raise SystemExit("eval subcommands: sim | parity")
        which, tail = rest[0], rest[1:]
        if which == "sim":
            from opentype_miner.eval.sim_duel import main as m
        elif which == "parity":
            from opentype_miner.eval.parity import main as m
        else:
            raise SystemExit(f"unknown eval command {which}")
        m(tail)
        return
    if head == "train":
        if not rest:
            raise SystemExit("train subcommands: reads | harness | wise-ft")
        which, tail = rest[0], rest[1:]
        if which == "reads":
            from opentype_miner.train.read_slot import main as m
        elif which == "harness":
            from opentype_miner.train.harness_sft import main as m
        elif which == "wise-ft":
            from opentype_miner.train.wise_ft import main as m
        else:
            raise SystemExit(f"unknown train command {which}")
        m(tail)
        return
    if head == "export":
        from opentype_miner.export.merge import main as m

        m(rest)
        return
    p.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
