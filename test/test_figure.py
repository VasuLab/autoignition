import autoignition

from numpy.testing import assert_array_equal
import pytest


class TestPropGroups:

    @staticmethod
    @pytest.fixture
    def figure():
        figure = autoignition.Figure()
        figure.prop_groups = {
            "red": dict(color=(1, 0, 0)),
            "black": dict(color=(0, 0, 0))
        }
        return figure

    @staticmethod
    def test_single_prop_group(figure):
        """Test that a property group will correctly set the color."""
        c = figure.plot_exp([], [], "red")
        assert_array_equal(c.get_facecolor(), [[1, 0, 0, 1]])  # red RGBA

    @staticmethod
    def test_multiple_prop_groups(figure):
        """Test that the property groups are applied in the expected order."""
        c = figure.plot_exp([], [], "red", "black")
        assert_array_equal(c.get_facecolor(), [[0, 0, 0, 1]])  # black RGBA

    @staticmethod
    def test_kwarg_override(figure):
        """Test that a keyword argument will override the property group."""
        c = figure.plot_exp([], [], "black", color=(0, 1, 0))
        assert_array_equal(c.get_facecolor(), [[0, 1, 0, 1]])  # green RGBA
