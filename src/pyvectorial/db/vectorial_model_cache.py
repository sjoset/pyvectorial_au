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

# globals - there has to be a better way to do this
vm_cache_engine: Optional[Engine] = None
vm_cache_session = None


class VMCachedBase(DeclarativeBase):
    pass


class VMCached(VMCachedBase):
    __tablename__ = "vmcache"
    vmc_hash: Mapped[str] = mapped_column(primary_key=True)
    vmr_b64_enc_zip: Mapped[bytes] = mapped_column(nullable=False)
    execution_time_s: Mapped[float] = mapped_column(nullable=False)
    vectorial_model_backend: Mapped[str] = mapped_column(nullable=False)
    vectorial_model_version: Mapped[str] = mapped_column(nullable=False)


def initialize_vectorial_model_cache(vectorial_model_cache_dir: pathlib.Path) -> None:
    global vm_cache_engine, vm_cache_session

    if vm_cache_engine is not None or vm_cache_session is not None:
        return

    full_db_path = vectorial_model_cache_dir / pathlib.Path("vmcache.sqlite3")
    vm_cache_engine = create_engine(f"sqlite:///{full_db_path}", poolclass=NullPool)

    VMCachedBase.metadata.create_all(bind=vm_cache_engine)

    session_factory = sessionmaker(vm_cache_engine)
    vm_cache_session = scoped_session(session_factory)


def get_vm_cache_db_session() -> Optional[Session]:
    global vm_cache_engine, vm_cache_session

    if vm_cache_session:
        return vm_cache_session()
    else:
        return None
