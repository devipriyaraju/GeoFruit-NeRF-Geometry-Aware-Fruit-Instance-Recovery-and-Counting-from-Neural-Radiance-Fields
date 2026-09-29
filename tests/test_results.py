from fruitnerf_repro.results import MAIN_RESULTS


def test_known_results():
    assert MAIN_RESULTS[0].ours == 168
    assert MAIN_RESULTS[1].ours == 98
