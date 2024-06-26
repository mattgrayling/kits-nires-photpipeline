import photutils
import astroquery
import numpy as np
import astropy.io.fits as fits

import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

import argparse

from astroquery.vizier import Vizier

Vizier.ROW_LIMIT = -1

from photutils.detection import DAOStarFinder
from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus, aperture_photometry
import copy, pdb


def query_cat(coord, size=10, catalog=None, verbose=False):
    """
    Query the 2MASS and UKIDSS catalogs for sources in the image.

    Input:
    coord: the SkyCoord of the object
    size : the query radius in arcmin, default is 10
    catalog: either "2MASS" or "UKIDSS", if None, use one with most objects
    """
    # Query 2MASS
    result_2m_raw = Vizier.query_region(SN_coord,
                                        width="%dm" % size,
                                        catalog=["II/246/out"])
    # weed out galaxies, is AAA needed?

    #
    # Query UKIDSS
    result_UKIDSS = Vizier.query_region(SN_coord,
                                        width="%dm" % size,
                                        catalog=["II/319/las9"])
    if verbose:
        print("#######2MASS RESULTS########")
        print(result_2m_raw)
        print("#######UKIDSS RESULTS########")
        print(result_UKIDSS)
        # print(len(result_UKIDSS))

    if (len(result_2m_raw) == 0) & (len(result_UKIDSS) == 0):
        print("No 2MASS or UKIDSS sources in this region. Tough luck!")
        return None

    # If only one is available
    elif len(result_2m_raw) == 0:
        print("Only UKIDSS available. Use UKIDSS.")
        result = result_UKIDSS
        return result
    elif len(result_UKIDSS) == 0:
        print("Only 2MASS available. Use 2MASS")
        # result_2m = result_2m_raw[0][ (result_2m_raw[0]['Qflg']=='AAA') & (result_2m_raw[0]['Bflg']=='111')]
        result = result_2m_raw[0]
        return result

    else:
        if catalog is None:
            if len(result_2m_raw[0]) >= len(result_UKIDSS[0]):
                # result_2m = result_2m_raw[0][ (result_2m_raw[0]['Qflg']=='AAA') & (result_2m_raw[0]['Bflg']=='111')]
                result = result_2m_raw[0]
                print('Use 2MASS')
            else:
                result = result_UKIDSS[0]
                print('Use UKIDSS')
        elif (catalog == 'UKIDSS'):
            result = result_UKIDSS[0]
            print('Use UKIDSS as requested')
        elif catalog == '2MASS':
            # result_2m = result_2m_raw[0][ (result_2m_raw[0]['Qflg']=='AAA') & (result_2m_raw[0]['Bflg']=='111')]
            result = result_2m_raw[0]
            print('Use 2MASS as requested')
        # elif len(result_2m) == 0:
        #   result = result_UKIDSS
        # elif len(result_UKIDSS) == 0:
        #   result = result_2m
        return result


def find_and_match_sources(image, wcs_header, standard_catalog, distance_pix=10, plot=False):
    """
    Run DAOStarFind to find sources in a given image
    Input:
        image: numpy array, should be background subtracted
        wcs: the image header with WCS information.
        standard_catalog: a catalog from query_cat
        distance_pix: how far can a source be away from the expected location.
                        This depends on the quality of your astrometry.
    Output:
        a catalog of all sources detected in the image
        a map between this catalog and the input standard catalog
    """
    # Compute stats to make threshold
    mean, med, std = sigma_clipped_stats(image)

    # Compute WCS
    wcs = WCS(wcs_header)

    # Find sources with 5 sigma detection
    # TO DO, measure seeting and update this somehow.
    daofind = DAOStarFinder(fwhm=5.0, threshold=5. * std)

    # Run daofind, these are sources found in the image.
    sources = daofind(image - med)

    # sources from the standard catalog to search
    std_coord = SkyCoord(ra=standard_catalog['RAJ2000'], \
                         dec=standard_catalog['DEJ2000'], unit=(u.deg, u.deg))
    # Convert to pixel coordinates
    std_pix = wcs.world_to_pixel(std_coord)  # + np.array([x_off, y_off])[:,np.newaxis]

    print("STANDARD CAT")
    print(std_pix)

    # Match sources based on distance
    matches = []

    for ind, i in enumerate(std_pix[0]):
        dist = np.sqrt((np.array(sources['xcentroid']) - std_pix[0][ind]) ** 2 +
                       (np.array(sources['ycentroid']) - std_pix[1][ind]) ** 2)
        idx = np.argmin(dist)  # Find a detected source closest to the standard target location
        if np.min(dist) < distance_pix:
            matches += [[idx, ind, np.min(dist), \
                         std_pix[0][ind] - sources[idx]['xcentroid'], std_pix[1][ind] - sources[idx]['ycentroid']]]
    matches = np.array(matches)

    if len(matches) == 0:
        print("WARNING: No matches found between detected source and catalog.")

    # PLOTTING SECTION
    if plot:

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(projection=wcs)
        ax.imshow(image, origin='lower', vmin=mean - 5 * std, vmax=mean + 5 * std)
        fig.tight_layout()

        # #SN aperture
        # SN_pix = wcs.world_to_pixel(SN_coord)
        # aperture_SN = CircularAperture(SN_pix, r=10)
        # ap_patches = aperture_SN.plot(color='blue', lw=2)

        # Show matched sources
        matches_px = []
        for ind, i in enumerate(matches):
            #     print(i[0])
            x, y = sources[int(i[0])]['xcentroid'], sources[int(i[0])]['ycentroid']
            matches_px += [[x, y]]

        # Show detected sources
        detected_xy = [[i['xcentroid'], i['ycentroid']] for i in sources]

        # print(detected_xy)

        ap_det = CircularAperture(detected_xy, r=10)
        ap_det_patches = ap_det.plot(color='green', lw=1)

        # Show catalog sources
        cat_xy = [[x[0], x[1]] for x in std_pix]

        ap_cat = CircularAperture(cat_xy, r=10)
        ap_cat_patches = ap_cat.plot(color='blue', lw=1)

        # Show matches
        ap_matches = CircularAperture(matches_px, r=10)
        ap_matches_patches = ap_matches.plot(color='white', lw=2)

        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')

        plt.show()

    # Note here that matches is a list of list where each element is:
    # [index of detected source, index of standard from catalog, distance, x_catalog-x_detected, y_catalog-y_detected]
    return sources, matches


def fix_WCS(fits_HDUList, sources, matches, standard_catalog, sip_degree=2):
    """
    Update the given WCS header with new WCS keys based on the standard stars
    Inputs:
        fits_HDUList: the whole HDU list of the image
        sources: source catalog from find_and_match_sources
        matches: matches catalog from find_and_match_sources
        standard_catalog: from from query_cat
        sip_degree: The degree of SIP to use.
    """
    from astropy.wcs.utils import fit_wcs_from_points

    # Make SkyCoord object
    std_coord = SkyCoord(ra=standard_catalog['RAJ2000'], \
                         dec=standard_catalog['DEJ2000'], unit=(u.deg, u.deg))

    # Fit for new wcs from soruces
    new_wcs = fit_wcs_from_points((sources[matches[:, 0].astype('int')]['xcentroid'],
                                   sources[matches[:, 0].astype('int')]['ycentroid']),
                                  std_coord[matches[:, 1].astype('int')], proj_point='center', projection='TAN',
                                  sip_degree=sip_degree)
    new_header = new_wcs.to_header(relax=True)

    # Update the given fits HDUList
    for ext in fits_HDU:
        for j in new_header:
            ext.header[j] = new_header[j]
    return updated_HDUList


def apPhot(image, variance_image, sources, radii=[1, 3, 5, 7, 9, 11, 15], bkg_an_in=1.5, bkg_an_out=3):
    """
    Compute aperture photometry for sources detected
    Input:
        image: numpy array containing the image
        variance_image: if available, a numpy array containing the variance
        sources: a source catalog from find_and_match_sources
    Output:
        a photometry table for all sources in the source table
    """

    # for i in good2M:
    ap_std = []
    bad_std = []
    # for i in sources:

    #     fake_ap = photutils.CircularAperture(pix, r=10)
    #     ap_stats = photutils.ApertureStats(final_im[1].data, fake_ap)
    # #     print(pix, ap_stats.centroid)
    # #     ap_UKIDSS += [ap_stats.centroid]
    # #     bad_UKIDSS += [pix]
    #     ap_std += [pix]

    sources_list = np.array([sources['xcentroid'], sources['ycentroid']]).T

    phot_tables = []

    for r in radii:
        aperture_source = CircularAperture(sources_list, r=r)
        annulus_source = CircularAnnulus(sources_list, r_in=r * bkg_an_in, r_out=r * bkg_an_out)

        # Calculate the background by taking a sigma clipped mean of the sky annulus
        aper_stats = ApertureStats(image, aperture_source, sigma_clip=None)
        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        bkg_stats = ApertureStats(image, annulus_source, sigma_clip=sigclip)
        total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value

        # photometry
        # aper_stats_bkgsub = ApertureStats(image, aperture_source,
        #                           local_bkg=bkg_stats.median)
        if variance_image is not None:
            phot_table = aperture_photometry(image, aperture_source, error=np.sqrt(variance_image))
        else:
            phot_table = aperture_photometry(image, aperture_source)
        # Add background subtraction
        phot_bkgsub = phot_table['aperture_sum'] - total_bkg
        phot_table['total_bkg'] = total_bkg
        phot_table['aperture_sum_bkgsub'] = phot_bkgsub

        # print(phot_table)

        # std_bkg_sub = aperture_source['aperture_sum'] - annulus_source['aperture_sum']/annulus_source.area*aperture_source.area
        # std_bkg_sub_err = np.sqrt(aperture_source['aperture_sum_err']**2 +
        #                 (annulus_source['aperture_sum_err']/annulus_source.area*aperture_source.area)**2)

        phot_tables += [phot_table]

        ############DO SOME CURVE OF GROWTH SHIT HERE##############

    return np.array(phot_tables)


def ZPCalculation(band, sources, standard_catalog, matches, phot_table, radii, plot=False):
    """
    Calculate the zeropoint using the raw photometry of standard stars in the image.
    Input:
        band: which filter it is. J, H, or K
        sources: sources table from query_cat
        standard_catalog: from find_and_match_sources
        matches: also from find_and_match_sources
        phot_table: the photometry table from apPhot
        plot: whether to make diagnostic plots

    Output:
        a zeropoint in this image with standard deviation.
        if selected, histogram of zeropoints and a plot showing ZP for different standard stars.
    """
    if band not in ['J', 'H', 'K', 'Ks']:
        print("Band must be J, H, K, or Ks. K and Ks are the same for now. Y is not supported")
        return None
    else:
        if band == 'Ks':
            band = 'K'

        # relavent photometry
        std_phot = phot_table[:, [int(x) for x in matches[:, 0]]]
        std_cat_mag = standard_catalog[[int(x) for x in matches[:, 1]]]  # matches should have stored this as int!

        # print(len(std_phot), len(std_cat_mag))
        if band == 'J':
            if 'Jmag' in std_cat_mag.colname:
                band_str = 'Jmag'
                band_err_str = 'e_Jmag'
            elif 'Jmag1' in std_cat_mag.colname:
                band_str = 'Jmag1'
                band_err_str = 'e_Jmag1'
            else:
                print('Uhhhh. Check the catalog?')
                print(std_cat_mag)
        else:
            band_str = band + 'mag'
            band_err_str = 'e_' + band + 'mag'

        ZP = std_cat_mag[band_str] + 2.5 * np.log10(std_phot[:]['aperture_sum_bkgsub'])
        ZP_err = np.sqrt(std_cat_mag[band_err_str] ** 2 + (
                    2.5 / std_phot[:]['aperture_sum_bkgsub'] / np.log(10) * std_phot[:]['aperture_sum_err']) ** 2)

        # Compute overall zeropoint

        final_ZP = []
        final_ZP_err = []
        for i in range(len(ZP)):
            mean, med, std = sigma_clipped_stats(ZP[i])
            final_ZP += [med]
            final_ZP_err += [std / np.sqrt(len(ZP[i]))]

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            for i, r in enumerate(radii):
                ax.hist(ZP[i], histtype='step', label=r)
                # ax.
            plt.legend()
            plt.show()
        final_ZP = np.array(final_ZP)
        final_ZP_err = np.array(final_ZP_err)

        return final_ZP, final_ZP_err


def computeMag(phot_table, ZP, ZP_err):
    if len(phot_table) != len(ZP):
        print('Photometry table and Zeropoints must be calculated using the same aperture radii')
    else:
        ############FIX THIS SO IT IS NOT LIMITED TO ONE SOURCE
        mag = -2.5 * np.log10(phot_table[:]['aperture_sum_bkgsub'][:, 0]) + ZP
        emag = np.sqrt(ZP_err ** 2 + (2.5 / phot_table[:]['aperture_sum_bkgsub'][:, 0] / \
                                      np.log(10) * phot_table[:]['aperture_sum_err'][:, 0]) ** 2)

        limit_3sig_SN = -2.5 * np.log10(3 * phot_table[:]['aperture_sum_err'][:, 0]) + ZP
        limit_5sig_SN = -2.5 * np.log10(5 * phot_table[:]['aperture_sum_err'][:, 0]) + ZP

    return mag, emag, limit_3sig_SN, limit_5sig_SN


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='NIR Photometry',
        description='A general code to perform NIR photometry on a final stacked NIR image with a decent WCS')
    # parser.add_argument('filename')
    # parser.add_argument('ra')
    # parser.add_argument('dec')
    parser.add_argument('band')
    parser.add_argument('-d', '--ext_data', default=0)
    parser.add_argument('-e', '--ext_var', default=1)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-m', '--max_dist', default=10)

    args = parser.parse_args()
    # print(args.ra, args.dec)

    SN_coord = SkyCoord('12:25:45.874 +12:39:48.87', unit = (u.hourangle, u.deg))
    image = fits.open('data/20240615/redux/combined/2023fyq_S1.fits.gz')

    # image = fits.open(args.filename)

    print(image)

    image_data = image[int(args.ext_data)].data
    plt.imshow(image_data)
    plt.show()
    # stats
    mean, med, std = sigma_clipped_stats(image_data)
    wcs = WCS(image[0].header)

    if args.ext_var == 'None':
        print("###Assuming var = data. Generally not true unless the counts are already converted to e-.###")
        image_error = image_data  # TERRIBLE ASSUMPTION
    else:
        print("###Squared this to get var###")
        image_error = image[int(args.ext_var)].data ** 2

    # FIX THIS TO BE MORE GENERAL

    if (':') in args.ra:
        SN_coord = SkyCoord(args.ra + ' ' + args.dec, unit=(u.hourangle, u.deg))
    else:
        SN_coord = SkyCoord(args.ra + ' ' + args.dec, unit=(u.deg, u.deg))

    print("Querying Vizier for 2MASS and UKIDSS stars.")
    std_cat = query_cat(SN_coord, size=10, catalog=None, verbose=args.verbose)
    if args.verbose:
        print(std_cat)

    print("Running source finding and catalog matching.")
    sources, matches = find_and_match_sources(image_data, image[0].header, std_cat, distance_pix=float(args.max_dist),
                                              plot=True)
    if args.verbose:
        print(matches)

        #########JUST FOR PLOTTING
        # fig = plt.figure(figsize = (10,10))
        # ax = plt.subplot(projection=wcs)
        # ax.imshow(image_data, origin = 'lower', vmin = mean-5*std, vmax = mean+5*std)
        # for i in matches:
        #     print(sources[int(i[0])])
        #     aperture = CircularAperture( [sources[int(i[0])]['xcentroid'],sources[int(i[0])]['ycentroid'] ], r=7)
        #     ap_patches = aperture.plot(color='white', lw=2)
        # fig.show()

    print("Performing photometry")
    radii = [1, 3, 5, 7, 9, 11, 15]
    ap_photo = apPhot(image_data, image_error, sources, radii=radii)
    print("Calculating zeropoint")
    # pdb.set_trace()
    final_ZP, final_ZP_err = ZPCalculation(args.band, sources, std_cat, matches, ap_photo, radii, plot=True)
    # print(final_ZP, final_ZP_err)

    # SN photometry
    ##########FIX THIS PART TO MAKE IT MORE GENERAL

    SN_pix = wcs.world_to_pixel(SN_coord) + np.array([-3, 0])
    SN_source = Table([[SN_pix[0]], [SN_pix[1]]], names=['xcentroid', 'ycentroid'])

    ########################################################
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=wcs)

    # THESE ARE JUST FOR PLOTTING
    aperture = CircularAperture(SN_pix, r=7)
    annulus = CircularAnnulus(SN_pix, r_in=7 * 1.5, r_out=7 * 3)

    ax.imshow(image_data, origin='lower', vmin=mean - 5 * std, vmax=mean + 70 * std)

    ax.set_xlim([SN_pix[0] - 100, SN_pix[0] + 100])
    ax.set_ylim([SN_pix[1] - 100, SN_pix[1] + 100])

    ap_patches = aperture.plot(color='white', lw=2)
    an_patches = annulus.plot(color='red', lw=2)
    plt.tight_layout()
    plt.show()
    ########################################################

    # SN_phot = apPhot(image[0].data, image[1].data,SN_source, radii = radii)

    # pdb.set_trace()

    SN_phot = apPhot(image_data, image_error, sources[[int(x) for x in matches[:, 0]]], radii=radii)

    # print(SN_phot)

    mag, emag, limit_3sig_SN, limit_5sig_SN = computeMag(SN_phot, final_ZP, final_ZP_err)

    # print(mag)
    # print(emag)
    outTable = Table([radii, mag, emag, limit_3sig_SN, limit_5sig_SN],
                     names=['radii_pix', 'mag', 'emag', 'mag_limit_3sig', 'mag_limit_5sig'])
    print(outTable)


