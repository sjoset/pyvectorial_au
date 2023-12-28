#!/usr/bin/env python3

import pathlib
import zlib

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# import pandas as pd


def get_db_engine(cache_dir_path: pathlib.Path):
    cache_path = cache_dir_path / pathlib.Path("vmcache.db")
    return create_engine(f"sqlite:///{cache_path}")


def get_db_connection(engine):
    return engine.connect()


def create_vmcache_tables(engine):
    VMCacheBase.metadata.create_all(bind=engine)


class VMCacheBase(DeclarativeBase):
    pass


class VMCache(VMCacheBase):
    __tablename__ = "vmcache"
    vmc_hash: Mapped[str] = mapped_column(primary_key=True)
    vmr_b64_enc_zip: Mapped[bytes] = mapped_column(nullable=False)
    execution_time_s: Mapped[float] = mapped_column(nullable=False)
    vectorial_model_backend: Mapped[str] = mapped_column(nullable=False)
    vectorial_model_version: Mapped[str] = mapped_column(nullable=False)
    # ver, execution time, etc


def compress_vmr_string(vmr: str) -> bytes:
    return zlib.compress(vmr.encode())


def decompress_vmr_string(vmr_compressed: bytes) -> str:
    return zlib.decompress(vmr_compressed).decode()


def get_example_vmcaches():
    vmc_hash = "a313ecd"
    vmr_b64 = "sanoidlsagoidoxuktodeuxtnhaoexuigdfaxuinthaxontidxanotheu"
    first_model = VMCache(
        vmc_hash=vmc_hash,
        vmr_b64_enc_zip=compress_vmr_string(vmr_b64),
        execution_time_s=1.0,
        vectorial_model_backend="python",
        vectorial_model_version="0.4.0",
    )
    second_model = VMCache(
        vmc_hash=vmr_b64,
        vmr_b64_enc_zip=compress_vmr_string(vmc_hash),
        execution_time_s=0.25,
        vectorial_model_backend="rust",
        vectorial_model_version="1.0.0",
    )

    return [first_model, second_model]


def get_new_model():
    vmc_hash = "lrcgsnthaoeu"
    vmr_b64 = "239560oeu90x8aoe08734xiktoeh"

    new_model = VMCache(
        vmc_hash=vmc_hash,
        vmr_b64_enc_zip=compress_vmr_string(vmr_b64),
        execution_time_s=0.5,
        vectorial_model_backend="fortran",
        vectorial_model_version="1.0.1",
    )

    return new_model


def add_some_entries(engine):
    with Session(engine) as session:
        session.add_all(get_example_vmcaches())
        session.commit()


def try_new_model(engine):
    new_model = get_new_model()
    test_hash = new_model.vmc_hash

    with Session(engine) as session:
        res = session.get(VMCache, test_hash)
        if res is None:
            print("inserting new model here")
            session.add(new_model)
            session.commit()
        else:
            print(f"found existing model for vmc hash {test_hash}")
            vmr = decompress_vmr_string(res.vmr_b64_enc_zip)
            print(f"vmr: {vmr}")


# def does_model_exist(vmc_hash, session: Session):
#     a = session.query(VMCache).filter_by(vmc_hash=vmc_hash).first()
#     if a is None:
#         return False
#     print(a.vmc_hash, a.vmr_base64_encoded, a.execution_time_s)
#     return True
#
#
# def get_cached_model(vmc_hash: str, session: Session):
#     a = session.query(VMCache).filter_by(vmc_hash=vmc_hash).first()
#     if a is not None:
#         print(a.vmc_hash, a.vmr_base64_encoded, a.execution_time_s)
#     return a


def main():
    engine = get_db_engine(pathlib.Path("."))
    create_vmcache_tables(engine)

    # add_some_entries(engine)

    with Session(engine) as session:
        x = session.get(VMCache, "a313ecd")
        if x is not None:
            print(
                f"Found entry with hash: {x.vmc_hash}",
                "val: ",
                decompress_vmr_string(x.vmr_b64_enc_zip),
            )

    try_new_model(engine)

    # df = pd.read_sql_table("vmcache", get_db_connection(engine))
    # print(df.execution_time_s)


if __name__ == "__main__":
    main()
