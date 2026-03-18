"""Tests for QAC factories — feature maps, ansatze, and optimizers."""
import pytest


class TestFeatureMapFactory:
    def test_zz_creation(self):
        from qac.feature_map_factory import FeatureMapFactory

        fm = FeatureMapFactory.create("zz", n_qubits=4)
        assert fm.num_qubits == 4

    def test_z_creation(self):
        from qac.feature_map_factory import FeatureMapFactory

        fm = FeatureMapFactory.create("z", n_qubits=4)
        assert fm.num_qubits == 4

    def test_exceeds_10_raises(self):
        from qac.feature_map_factory import FeatureMapFactory

        with pytest.raises(ValueError, match="≤ 10"):
            FeatureMapFactory.create("zz", n_qubits=12)

    def test_unknown_type_raises(self):
        from qac.feature_map_factory import FeatureMapFactory

        with pytest.raises(ValueError, match="Unknown"):
            FeatureMapFactory.create("unknown", n_qubits=4)


class TestAnsatzFactory:
    def test_real_amplitudes(self):
        from qac.ansatz_factory import AnsatzFactory

        ansatz = AnsatzFactory.create("real_amplitudes", n_qubits=4)
        assert ansatz.num_qubits == 4
        assert ansatz.num_parameters > 0

    def test_efficient_su2(self):
        from qac.ansatz_factory import AnsatzFactory

        ansatz = AnsatzFactory.create("efficient_su2", n_qubits=4)
        assert ansatz.num_qubits == 4

    def test_unknown_raises(self):
        from qac.ansatz_factory import AnsatzFactory

        with pytest.raises(ValueError, match="Unknown"):
            AnsatzFactory.create("unknown", n_qubits=4)


class TestOptimizerFactory:
    def test_cobyla(self):
        from qac.optimizer_factory import OptimizerFactory

        opt = OptimizerFactory.create("cobyla", max_iter=50)
        assert opt is not None

    def test_spsa(self):
        from qac.optimizer_factory import OptimizerFactory

        opt = OptimizerFactory.create("spsa", max_iter=50)
        assert opt is not None

    def test_unknown_raises(self):
        from qac.optimizer_factory import OptimizerFactory

        with pytest.raises(ValueError, match="Unknown"):
            OptimizerFactory.create("adam", max_iter=50)
