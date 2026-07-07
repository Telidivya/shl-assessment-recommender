import json

import pytest

from app.catalog import Catalog
from app.exceptions import CatalogLoadError


def test_loads_valid_catalog_from_default_path():
    catalog = Catalog()
    assert len(catalog) > 0
    assert all(p.url.startswith("https://www.shl.com/") for p in catalog.products)


def test_missing_file_raises_catalog_load_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(CatalogLoadError):
        Catalog(path=missing_path)


def test_malformed_json_raises_catalog_load_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json")
    with pytest.raises(CatalogLoadError):
        Catalog(path=bad_file)


def test_missing_products_key_raises_catalog_load_error(tmp_path):
    bad_file = tmp_path / "no_products.json"
    bad_file.write_text(json.dumps({"_meta": {}}))
    with pytest.raises(CatalogLoadError):
        Catalog(path=bad_file)


def test_empty_products_list_raises_catalog_load_error(tmp_path):
    bad_file = tmp_path / "empty.json"
    bad_file.write_text(json.dumps({"products": []}))
    with pytest.raises(CatalogLoadError):
        Catalog(path=bad_file)


def test_entry_missing_required_field_raises_catalog_load_error(tmp_path):
    bad_file = tmp_path / "missing_field.json"
    bad_file.write_text(json.dumps({"products": [{"name": "X"}]}))  # missing "url"
    with pytest.raises(CatalogLoadError):
        Catalog(path=bad_file)


def test_duration_minutes_is_derived_from_description_when_not_set():
    bad_file_products = {
        "products": [
            {
                "name": "Sample Test",
                "url": "https://www.shl.com/products/product-catalog/view/sample/",
                "test_type": ["K"],
                "description": "A short test that takes about 20 minutes to complete.",
            }
        ]
    }
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "catalog.json"
        path.write_text(json.dumps(bad_file_products))
        catalog = Catalog(path=path)
        assert catalog.products[0].duration_minutes == 20
