import pathlib
from sqlalchemy import NullPool, create_engine
from pyvectorial_au.db.vectorial_model_cache import VMCachedBase


def initialize_vectorial_model_cache(
    vectorial_model_cache_dir: pathlib.Path,
    vectorial_model_cache_filename: pathlib.Path = pathlib.Path("vmcache.sqlite3"),
) -> pathlib.Path:
    """
    On startup we call this before any work is done to make sure the db has the proper
    tables and columns, and return the full file path for use later
    """

    full_db_path = vectorial_model_cache_dir.joinpath(vectorial_model_cache_filename)
    vm_cache_engine = create_engine(f"sqlite:///{full_db_path}", poolclass=NullPool)

    VMCachedBase.metadata.create_all(bind=vm_cache_engine)

    return full_db_path
