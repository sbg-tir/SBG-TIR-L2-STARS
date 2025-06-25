import subprocess
import logging

logger = logging.getLogger(__name__)

def install_STARS_jl(
    github_URL: str = "https://github.com/STARS-Data-Fusion/STARS.jl",
    environment_name: str = "@SBGv001-L2T-STARS") -> subprocess.CompletedProcess:
    """
    Installs the STARS.jl Julia package from GitHub into a specified Julia environment.

    This function executes a Julia command to activate a given environment and
    then develops (installs in editable mode) the STARS.jl package from its
    GitHub repository.

    Args:
        github_URL (str, optional): The URL of the GitHub repository containing STARS.jl.
                                    Defaults to "https://github.com/STARS-Data-Fusion/STARS.jl".
        environment_name (str, optional): The name of the Julia environment to install
                                          the package into. Defaults to "@SBGv001-L2T-STARS".

    Returns:
        subprocess.CompletedProcess: An object containing information about the
                                     execution of the Julia command (return code, stdout, stderr).
    """
    # Julia command to activate an environment and then add/develop a package from URL
    julia_command = [
        "julia",
        "-e",
        f'using Pkg; Pkg.activate("{environment_name}"); Pkg.develop(url="{github_URL}")',
    ]

    # Execute the Julia command as a subprocess
    result = subprocess.run(julia_command, capture_output=True, text=True, check=False)

    if result.returncode == 0:
        logger.info(
            f"STARS.jl installed successfully in environment '{environment_name}'!"
        )
    else:
        logger.error("Error installing STARS.jl:")
        logger.error(result.stderr)
    return result
