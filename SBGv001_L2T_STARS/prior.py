from datetime import date

class Prior:
    """
    A data class to encapsulate information about a prior L2T_STARS product.

    This class holds filenames and flags related to the use of a previous
    STARS product as a 'prior' in the data fusion process. This can help
    to constrain the solution and improve accuracy, especially when
    observations for the current date are sparse.

    Attributes:
        using_prior (bool): True if a prior product is being used, False otherwise.
        prior_date_UTC (date): The UTC date of the prior product.
        L2T_STARS_prior_filename (str): Path to the prior L2T_STARS zip file.
        prior_NDVI_filename (str): Path to the prior NDVI mean file.
        prior_NDVI_UQ_filename (str): Path to the prior NDVI uncertainty (UQ) file.
        prior_NDVI_flag_filename (str): Path to the prior NDVI flag file.
        prior_NDVI_bias_filename (str): Path to the prior NDVI bias file.
        prior_NDVI_bias_UQ_filename (str): Path to the prior NDVI bias uncertainty file.
        prior_albedo_filename (str): Path to the prior albedo mean file.
        prior_albedo_UQ_filename (str): Path to the prior albedo uncertainty (UQ) file.
        prior_albedo_flag_filename (str): Path to the prior albedo flag file.
        prior_albedo_bias_filename (str): Path to the prior albedo bias file.
        prior_albedo_bias_UQ_filename (str): Path to the prior albedo bias uncertainty file.
    """

    def __init__(
        self,
        using_prior: bool = False,
        prior_date_UTC: date = None,
        L2T_STARS_prior_filename: str = None,
        prior_NDVI_filename: str = None,
        prior_NDVI_UQ_filename: str = None,
        prior_NDVI_flag_filename: str = None,
        prior_NDVI_bias_filename: str = None,
        prior_NDVI_bias_UQ_filename: str = None,
        prior_albedo_filename: str = None,
        prior_albedo_UQ_filename: str = None,
        prior_albedo_flag_filename: str = None,
        prior_albedo_bias_filename: str = None,
        prior_albedo_bias_UQ_filename: str = None,
    ):
        self.using_prior = using_prior
        self.prior_date_UTC = prior_date_UTC
        self.L2T_STARS_prior_filename = L2T_STARS_prior_filename
        self.prior_NDVI_filename = prior_NDVI_filename
        self.prior_NDVI_UQ_filename = prior_NDVI_UQ_filename
        self.prior_NDVI_flag_filename = prior_NDVI_flag_filename
        self.prior_NDVI_bias_filename = prior_NDVI_bias_filename
        self.prior_NDVI_bias_UQ_filename = prior_NDVI_bias_UQ_filename
        self.prior_albedo_filename = prior_albedo_filename
        self.prior_albedo_UQ_filename = prior_albedo_UQ_filename
        self.prior_albedo_flag_filename = prior_albedo_flag_filename
        self.prior_albedo_bias_filename = prior_albedo_bias_filename
        self.prior_albedo_bias_UQ_filename = prior_albedo_bias_UQ_filename
