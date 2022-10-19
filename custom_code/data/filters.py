"""Common filters to exclude data."""

from loguru import logger

import math

def filter_out_noise(container,story):
    """Filter out data with noise.

    Parameters
    ----------
    container: OmsiContainer
        Container to apply filtering on

    Returns
    -------
    bool
        True if the stimulus of container was NOT presented in noise.
    """
    return_value = True
    snr_found = False
    z = container['metadata']

    if "snr" in container[z[0][0]].keys():
        if math.isnan(container[z[0][0]]['snr'][0,0]):
            is_snr_nan = True
            return_value = return_value and (is_snr_nan)
        else:
            is_snr_100 = int(container[z[0][0]]['snr'][0,0]) == 100
            is_snr_1 = int(container[z[0][0]]['snr'][0,0]) == 1
            return_value = return_value and (is_snr_100 or is_snr_1)

    try:
        snr = int(
            container["subject/"+story+"/test/snr"][0,0]
        )
        return_value = return_value and snr == 100
        snr_found = True
    except KeyError:
        pass

#    if not snr_found:
#        logger.info("No snr found for container %s" % container.path)

    return return_value