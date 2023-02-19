from module.scripts.splitmerge.merge import _get_params, SliceParams


def test_simple_get_params():
    name = "lena_win_1_2_sh_11_12_pos_21_22.jpg"
    ans = {
        "win_shape": (1, 2),
        "shift_shape": (11, 12),
        "pos": (21, 22),
    }

    assert _get_params(name) == SliceParams(**ans)