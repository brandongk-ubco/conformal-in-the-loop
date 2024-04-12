from citl.ConformalClassifier import ConformalClassifier


class TestConformalClassifier:
    def test_initialize(self):
        cc = ConformalClassifier()
        assert cc.__sklearn_is_fitted__() == True
        assert len(cc.cp_examples) == 0
        assert cc.mapie_classifier == None

