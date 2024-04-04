import pathlib
from typing import Optional

from sqlalchemy import Engine, NullPool, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    scoped_session,
    sessionmaker,
)

"""
From the sqlalchemy docs:
    When using an Engine with multiple Python processes,
    such as when using os.fork or Python multiprocessing,
    it’s important that the engine is initialized per process.
"""


class VMCachedBase(DeclarativeBase):
    pass


class VMCached(VMCachedBase):
    __tablename__ = "vmcache"
    vmc_hash: Mapped[str] = mapped_column(primary_key=True)
    vmr_b64_enc_zip: Mapped[bytes] = mapped_column(nullable=False)
    execution_time_s: Mapped[float] = mapped_column(nullable=False)
    vectorial_model_backend: Mapped[str] = mapped_column(nullable=False)
    vectorial_model_version: Mapped[str] = mapped_column(nullable=False)


def get_vectorial_model_cache_engine(
    vectorial_model_cache_path: pathlib.Path,
) -> Engine:
    """
    Returns an engine to the db based on the given path
    """

    vm_cache_engine = create_engine(
        f"sqlite:///{vectorial_model_cache_path}", poolclass=NullPool
    )

    return vm_cache_engine


def get_vectorial_model_cache_session(
    vectorial_model_cache_path: pathlib.Path,
    cache_engine: Optional[Engine] = None,
) -> Optional[Session]:
    """
    Returns a db session based on the given Engine, but if none was given we construct
    an engine with the given path.
    If an engine is specified, the path is ignored.
    """

    if not cache_engine:
        cache_engine = get_vectorial_model_cache_engine(
            vectorial_model_cache_path=vectorial_model_cache_path,
        )

    session_factory = sessionmaker(cache_engine)
    cache_session = scoped_session(session_factory)
    return cache_session()
