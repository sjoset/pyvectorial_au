import pathlib
from typing import Optional

from pyvectorial_au.db.vectorial_model_cache import (
    get_vectorial_model_cache_session,
    VMCached,
)
from pyvectorial_au.model_input.vectorial_model_config import VectorialModelConfig
from pyvectorial_au.model_output.vectorial_model_calculation import EncodedVMCalculation
from pyvectorial_au.encoding.encoding_and_hashing import (
    vmc_to_sha256_digest,
    decompress_vmr_string,
    compress_vmr_string,
)


def get_saved_model_from_cache(
    vmc: VectorialModelConfig,
    vectorial_model_cache_path: pathlib.Path,
) -> Optional[EncodedVMCalculation]:
    """
    Given a vectorial model config, look up in the database at vectorial_model_cache_path
    whether that config has been run, and return the already-run model if so.
    """

    hashed_vmc = vmc_to_sha256_digest(vmc=vmc)
    # print(f"Looking up hashed vmc {hashed_vmc} in db ...")
    vm_cache_session = get_vectorial_model_cache_session(
        vectorial_model_cache_path=vectorial_model_cache_path
    )
    if vm_cache_session is None:
        print(f"No db session, skipping. [hash {hashed_vmc}]")
        return None

    cache_result = vm_cache_session.get(VMCached, hashed_vmc)
    if not cache_result:
        # print(f"Not found in db. [hash {hashed_vmc}]")
        vm_cache_session.close()
        return None

    evmc = EncodedVMCalculation(
        vmc=vmc,
        evmr=decompress_vmr_string(cache_result.vmr_b64_enc_zip),
        execution_time_s=cache_result.execution_time_s,
        vectorial_model_backend=cache_result.vectorial_model_backend,
        vectorial_model_version=cache_result.vectorial_model_version,
    )
    vm_cache_session.close()
    return evmc


def save_model_to_cache(
    evmc: EncodedVMCalculation, vectorial_model_cache_path: pathlib.Path
) -> None:
    """
    Given a completed model, check the given database at vectorial_model_cache_path if the given model
    already exists in the database.  If it does not, add it.
    """
    hashed_vmc = vmc_to_sha256_digest(vmc=evmc.vmc)
    # print(f"Saving model with hashed vmc {hashed_vmc} in db ...")

    vm_cache_session = get_vectorial_model_cache_session(
        vectorial_model_cache_path=vectorial_model_cache_path
    )
    if vm_cache_session is None:
        print(f"No db session, skipping. [hash {hashed_vmc}]")
        return

    with vm_cache_session.begin():
        model_exists = (
            vm_cache_session.query(VMCached).filter_by(vmc_hash=hashed_vmc).first()
        )
        if not model_exists:
            to_db = VMCached(
                vmc_hash=hashed_vmc,
                vmr_b64_enc_zip=compress_vmr_string(evmc.evmr),
                execution_time_s=evmc.execution_time_s,
                vectorial_model_backend=evmc.vectorial_model_backend,
                vectorial_model_version=evmc.vectorial_model_version,
            )
            vm_cache_session.add(to_db)
            vm_cache_session.commit()
