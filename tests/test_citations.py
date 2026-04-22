import unittest
import cogsworth
import os


class Test(unittest.TestCase):
    def test_cite_galaxy(self):
        """Test citations for Galaxy"""
        g = cogsworth.sfh.Wagg2022()
        g.get_citations(filename="test.bib")

        self.assertTrue(os.path.exists("test.bib"))
        os.remove("test.bib")

    def test_cite_galaxy_no_bibs(self):
        """Test citations for Galaxy with no bibs"""
        class DummyGalaxy(cogsworth.sfh.StarFormationHistory):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.__citations__ = []
        g = DummyGalaxy()
        g.get_citations(filename="test.bib")

    def test_cite_pop(self):
        """Test citations for Galaxy"""
        p = cogsworth.pop.Population(5, processes=6, use_default_BSE_settings=True)
        p.create_population(with_timing=False)
        p.get_citations(filename="test.bib")

        self.assertTrue(os.path.exists("test.bib"))
        os.remove("test.bib")

    def test_utils_dependencies(self):
        """Test that the utils module imports all dependencies correctly"""
        import cogsworth.utils

        it_worked = True
        try:
            cogsworth.utils.check_dependencies("fake_package")
        except AttributeError:
            it_worked = False
            
        self.assertFalse(it_worked, "check_dependencies correctly raised an AttributeError for fake package")


def test_cite_stdin_pop(monkeypatch):
    """Test citations when using stdin/stdout (monkeypatch time!)"""
    monkeypatch.setattr('builtins.input', lambda _: "")

    p = cogsworth.pop.Population(5, processes=6, use_default_BSE_settings=True)
    p.create_population(with_timing=False)
    p.get_citations(filename=None)


def test_cite_stdin_galaxy(monkeypatch):
    """Test citations when using stdin/stdout (monkeypatch time!)"""
    monkeypatch.setattr('builtins.input', lambda _: "")

    g = cogsworth.sfh.Wagg2022()
    g.get_citations(filename=None)
