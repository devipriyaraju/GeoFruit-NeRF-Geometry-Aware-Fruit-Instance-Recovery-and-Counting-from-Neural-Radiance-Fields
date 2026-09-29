from statistics import mean

from fruitnerf_repro.results import MAIN_RESULTS


def main():
    print("tree     gt  paper  ours  paper_abs_err  ours_abs_err  paper_pct  ours_pct")
    for r in MAIN_RESULTS:
        print(
            f"{r.tree:8s} {r.gt:3d} {r.paper:6d} {r.ours:5d} "
            f"{abs(r.paper_error):13d} {abs(r.ours_error):12d} "
            f"{100 * r.paper_ratio:8.1f} {100 * r.ours_ratio:8.1f}"
        )
    paper_mae = mean(abs(r.paper_error) for r in MAIN_RESULTS)
    ours_mae = mean(abs(r.ours_error) for r in MAIN_RESULTS)
    print(f"mean absolute error, paper: {paper_mae:.1f}")
    print(f"mean absolute error, ours: {ours_mae:.1f}")


if __name__ == "__main__":
    main()
