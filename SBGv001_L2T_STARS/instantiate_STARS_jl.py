import subprocess
import logging

logger = logging.getLogger(__name__)

def instantiate_STARS_jl(package_location: str) -> subprocess.CompletedProcess:
    """
    Activates a Julia project at a given location and instantiates its dependencies.

    This is necessary to ensure all required Julia packages for STARS.jl are
    downloaded and ready for use within the specified project environment.

    Args:
        package_location (str): The directory of the Julia package (where Project.toml is located)
                                to activate and instantiate.

    Returns:
        subprocess.CompletedProcess: An object containing information about the
                                     execution of the Julia command (return code, stdout, stderr).
    """
    # Julia command to activate a specific package location and then instantiate its dependencies
    julia_command = [
        "julia",
        "-e",
        f'using Pkg; Pkg.activate("{package_location}"); Pkg.instantiate()',
    ]

    # Execute the Julia command as a subprocess
    result = subprocess.run(julia_command, capture_output=True, text=True, check=False)

    if result.returncode == 0:
        logger.info(
            f"STARS.jl instantiated successfully in directory '{package_location}'!"
        )
    else:
        logger.error("Error instantiating STARS.jl:")
        logger.error(result.stderr)
    return result
