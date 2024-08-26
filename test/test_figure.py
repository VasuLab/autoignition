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
        assert figure.plot_exp([], [], "red").lines[0]._color == (1, 0, 0)

    @staticmethod
    def test_multiple_prop_groups(figure):
        """Test that the property groups are applied in the expected order."""
        assert figure.plot_exp([], [], "red", "black").lines[0]._color == (0, 0, 0)

    @staticmethod
    def test_kwarg_override(figure):
        """Test that a keyword argument will override the property group."""
        assert figure.plot_exp([], [], "black", color=(0, 1, 0)).lines[0]._color == (0, 1, 0)
