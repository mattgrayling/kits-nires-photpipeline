import os, glob, copy

import astropy.stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import image_registration
import cv2

mpl.use('macosx')
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table, vstack
from astropy.stats import sigma_clip
from drizzle.drizzle import Drizzle
import photutils
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture, CircularAnnulus
import astroalign as aa
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astroquery.vizier import Vizier
from tqdm import tqdm
from copy import deepcopy

# We will divide by zero. Shoot me.
np.seterr(divide='ignore')


def makeFlat(image_cube, header_list, min_exp_time=5, sigma=5, \
             combine_method='mean', bpm_threshold=5, write_flat_name=None):
    """
    Create a master flat image from a given numpy array containing all images
    from a given night and a list of headers.

    The script normalize each image by its exposure time and combine them using
    sigma clipping. User can choose whether to use mean or median and the sigma
    used to clip the data.

    Input:
        image_cube:     3d numpy array containing all images from the night
        header_list:    a list of the same length as image_cube's 0th dimension containing headers
        min_exp_time:   the mininum exposure time of images used to make flat frame
        sigma:          sigma used by the sigma clipping algorithem to combine images
        combine_method: either 'mean' or 'median'
        bpm_threshold:  how many sigma away from average to mask a pixel as bad
        write_flat_name: path and name to output the master flat. If None, no file is written
    Output:
        masterFlat:     the master flat image, normalized so the median is unity
        bpm_map:        a boolean map of bad pixels
    """
    # Collect exposure time from the header
    exp_times = []
    for i in header_list:
        exp_times += [i['ITIME'] * i['COADDS']]
    exp_times = np.array(exp_times)
    # normalize by exp time
    image_cube = image_cube / exp_times[..., None, None]
    # Select only exposures with long time
    good = exp_times > min_exp_time
    # Compute combined images
    mean, med, std = sigma_clipped_stats(image_cube[good], axis=0, sigma=sigma)
    # Bad pixel map
    if combine_method == 'mean':
        combined = mean
    elif combine_method == 'median':
        combined = med
    overall_median = np.nanmedian(combined)
    overall_std = np.nanstd(std)

    flat = combined / overall_median
    # MG - adding back for now-----
    flat[flat < 0.5] = 1
    # -----------------------------
    # Bad Pixel Mask, True is bad
    bpm = (combined < overall_median - bpm_threshold * overall_std) & \
          (combined > overall_median + bpm_threshold * overall_std)

    if write_flat_name is not None:
        # TO DO write more metadata to these FITS
        header_to_write = copy.deepcopy(header_list[0])
        header_to_write['OBJECT'] = 'SKYFLAT'
        header_to_write['HISTORY'] = 'Master Flat for %s' % (header_to_write['DATE-OBS'])
        header_to_write['HISTORY'] = 'Files included:'
        for ind, hdr in enumerate(header_list):
            if good[ind]:
                header_to_write['HISTORY'] = hdr['DATAFILE']
        hdu = fits.PrimaryHDU(flat, header=header_to_write)
        hdu_mask = fits.ImageHDU(bpm.astype(int))
        hdul = fits.HDUList([hdu, hdu_mask])
        print(write_flat_name)
        hdul.writeto(write_flat_name, overwrite=True)

    return flat, bpm  # Return the normalized flat and the bad pixel map


def makeBackground(image_cube):
    """

    """


def fix_NIRES_WCS(header):
    """
    Fix NIRES SVC WCS header. The WCS header provided with NIRES SVC data
    does NOT take the PA into account. We fix this by rotating the CD matrix
    by 180 - PA degrees.

    Input: original header
    Output: FITS header with the CD matrix fixed
    """

    # if header['TARGNAME'] == '2023fyq':
    #     header['ROTDEST'] = 58
    #     header['PA'] = 58

    PA = header['ROTDEST']  # PA of vertical axis of the image

    # Hack fix for 2023fyq
    # if header['TARGNAME'] == '2023fyq':
    #     PA = 58.0
    #     header['PA'] = 58

    # load up the original CD
    original_CD = np.array([[header['CD1_1'], header['CD1_2']],
                            [header['CD2_1'], header['CD2_2']]])
    theta = np.radians(180 - PA)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], \
                                [np.sin(theta), np.cos(theta)]])

    new_CD = np.matmul(rotation_matrix, original_CD)

    header['CD1_1'] = new_CD[0, 0]
    header['CD1_2'] = new_CD[0, 1]
    header['CD2_1'] = new_CD[1, 0]
    header['CD2_2'] = new_CD[1, 1]

    return header

def query_cat(coord, size = 10, catalog = None, verbose = False):
    """
    Query the 2MASS and UKIDSS catalogs for sources in the image.

    Input:
    coord: the SkyCoord of the object
    size : the query radius in arcmin, default is 10
    catalog: either "2MASS" or "UKIDSS", if None, use one with most objects
    """
    #Query 2MASS
    result_2m_raw = Vizier.query_region(coord,
                        width="%dm"%size,
                        catalog=["II/246/out"])
    #weed out galaxies, is AAA needed?

    #
    #Query UKIDSS
    result_UKIDSS = Vizier.query_region(coord,
                        width="%dm"%size,
                        catalog=["II/319/las9"])
    if verbose:
        print("#######2MASS RESULTS########")
        print(result_2m_raw)
        print("#######UKIDSS RESULTS########")
        print(result_UKIDSS)
        # print(len(result_UKIDSS))

    if (len(result_2m_raw) == 0) & (len(result_UKIDSS)==0):
        print("No 2MASS or UKIDSS sources in this region. Tough luck!")
        return None

    #If only one is available
    elif len(result_2m_raw) == 0:
        print("Only UKIDSS available. Use UKIDSS.")
        result = result_UKIDSS
        return result
    elif len(result_UKIDSS)==0:
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

def find_and_match_sources(image, wcs_header, standard_catalog, distance_pix=50, plot=False):
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
    for x, y in zip(std_pix[0], std_pix[1]):
        print(x, y)

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
        cat_xy = [[x, y] for x, y in zip(std_pix[0], std_pix[1])]

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

def fix_WCS(fits_HDUList, sources, matches, standard_catalog, sip_degree = 2):
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

    #Make SkyCoord object
    std_coord = SkyCoord(ra = standard_catalog['RAJ2000'],\
        dec = standard_catalog['DEJ2000'], unit = (u.deg, u.deg))

    #Fit for new wcs from soruces
    new_wcs = fit_wcs_from_points( (sources[matches[:,0].astype('int')]['xcentroid'],
                                    sources[matches[:,0].astype('int')]['ycentroid']),
                                  std_coord[matches[:,1].astype('int')], proj_point='center', projection='TAN',
                                  sip_degree=sip_degree)
    new_header = new_wcs.to_header(relax=True)

    #Update the given fits HDUList
    for ext in fits_HDU:
        for j in new_header:
            ext.header[j] = new_header[j]
    return updated_HDUList

def apPhot(image, variance_image, sources, radii = [1,3,5,7,9,11,15], bkg_an_in = 1.5, bkg_an_out = 3):
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
        annulus_source = CircularAnnulus(sources_list, r_in=r*bkg_an_in, r_out = r*bkg_an_out)

        #Calculate the background by taking a sigma clipped mean of the sky annulus
        plt.imshow(image, vmin=1000, vmax=20000)
        aper_stats = ApertureStats(image, aperture_source, sigma_clip=None)
        sigclip = SigmaClip(sigma=3.0, maxiters=10)
        bkg_stats = ApertureStats(image, annulus_source, sigma_clip=sigclip)
        total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value

        #photometry
        # aper_stats_bkgsub = ApertureStats(image, aperture_source,
        #                           local_bkg=bkg_stats.median)
        phot_table = aperture_photometry(image, aperture_source, error=np.sqrt(variance_image))
        #Add background subtraction
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

def ZPCalculation(band, sources, standard_catalog, matches, phot_table, radii, plot = False):
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

        #relavent photometry
        std_phot = phot_table[:,[int(x) for x in matches[:,0]]]
        std_cat_mag = standard_catalog[[int(x) for x in matches[:,1]]] #matches should have stored this as int!

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
            band_str = band+'mag'
            band_err_str = 'e_'+band+'mag'

        ZP = std_cat_mag[band_str] + 2.5*np.log10(std_phot[:]['aperture_sum_bkgsub'])
        ZP_err = np.sqrt(std_cat_mag[band_err_str]**2 + (2.5/std_phot[:]['aperture_sum_bkgsub']/np.log(10)*std_phot[:]['aperture_sum_err'])**2)

        #Compute overall zeropoint

        final_ZP = []
        final_ZP_err = []
        for i in range(len(ZP)):
            mean, med, std = sigma_clipped_stats(ZP[i])
            final_ZP += [mean]
            final_ZP_err += [std/np.sqrt(len(ZP[i]))]

        if plot:
            fig, ax = plt.subplots(1,1,figsize = (6,4))
            for i, r in enumerate(radii):
                ax.hist(ZP[i], histtype = 'step', label = r)
                # ax.
            plt.legend()
            plt.show()
        final_ZP = np.array(final_ZP)
        final_ZP_err = np.array(final_ZP_err)

        return final_ZP, final_ZP_err

def computeMag(phot_table, ZP, ZP_err):

    if len(phot_table)!= len(ZP):
        print('Photometry table and Zeropoints must be calculated using the same aperture radii')
    else:
        ############FIX THIS SO IT IS NOT LIMITED TO ONE SOURCE
        mag = -2.5*np.log10(phot_table[:]['aperture_sum_bkgsub'][:,0]) + ZP
        emag = np.sqrt(ZP_err**2 + (2.5/phot_table[:]['aperture_sum_bkgsub'][:,0]/\
            np.log(10)*phot_table[:]['aperture_sum_err'][:,0])**2)

        limit_3sig_SN = -2.5*np.log10(3*phot_table[:]['aperture_sum_err'][:,0]) + ZP
        limit_5sig_SN = -2.5*np.log10(5*phot_table[:]['aperture_sum_err'][:,0]) + ZP

    return mag, emag, limit_3sig_SN, limit_5sig_SN


def reduceNight(directory, write_flat=True, write_bkg=True,
                write_flatten=True, write_bkg_sub=True,
                verbose=True, alignment_method='image_registration'):
    """
    This is the main function used to reduce an entire night of data.
    Input:
        directory:      string of the path to the raw NIRES SVC images
        write_flat:     bool write master flat for the night
        write_bkg:      bool write background frame for each target
        write_flatten:  bool write flat-corrected science images
        write_bkg_sub:  bool write the individual background subtracted sci images
        alignment_method: str method to use for fine image alignment, must be one of [ref_source, image_registration, cross_correlation]
    Output:
        Reduced files are written in the subdirectories of the given path
        By default, the master flat, the background for each target, and the stacked
        science image for each targets are written.
    """
    # Create subdirectories to write output files

    if not os.path.exists(f'data/{directory}/flat/'):
        os.mkdir(f'data/{directory}/flat/')
    if not os.path.exists(f'data/{directory}/redux/'):
        os.mkdir(f'data/{directory}/redux/')
    if not os.path.exists(f'data/{directory}/redux/combined/'):
        os.mkdir(f'data/{directory}/redux/combined/')
    if not os.path.exists(f'data/{directory}/redux/fixed_astrometry/'):
        os.mkdir(f'data/{directory}/redux/fixed_astrometry/')
    if not os.path.exists(f'data/{directory}/redux/background/'):
        os.mkdir(f'data/{directory}/redux/background/')

        # Find all NIRES files
    file_list = glob.glob('data/' + directory + '/v*fits*')
    file_list.sort()
    file_list = np.array(file_list)
    if len(file_list) == 0:
        print('No FITS files found in the given path: %s' % directory)

    if verbose:
        print(file_list)

    # TO DO filter out files that are not NIRES, other invalid files.

    # Load all data for given night--------------------------------------------
    all_targets = []
    all_imgs = []
    all_flats = []
    all_heads = []

    # Reading in all files for  night
    for i, file in tqdm(enumerate(file_list), total=len(file_list)):
        with fits.open(file) as hdu:
            head = hdu[0].header
            img = hdu[0].data
        target = head['TARGNAME']
        print(file, target)
        all_imgs.append(img)
        all_heads.append(fix_NIRES_WCS(head))
        all_targets.append(target)
    all_targets = np.array(all_targets)
    all_imgs = np.array(all_imgs)

    # ##FIX WCS FOR ALL HEADERS
    # all_heads = copy.deepcopy(old_all_heads)
    # for ind in range(len(all_heads)):
    #     all_heads[ind] = fix_NIRES_WCS(old_all_heads[ind])

    ########################################FLAT FIELDING######################################################
    # Call a function to make flat frame
    if write_flat:
        flat, bpm = makeFlat(all_imgs, all_heads, write_flat_name=f'data/{directory}/flat/{directory}_flat.fits')
    else:
        flat, bpm = makeFlat(all_imgs, all_heads, write_flat_name=None)

    # Flatten all images
    all_imgs /= flat[None, ...]

    # Write flatted files if requested
    if write_flatten:
        for i, file in enumerate(file_list):
            out_name = f'data/{directory}/redux/flatten_' + file.split('/')[-1]
            #     print(out_name)
            hdu = fits.open(file)
            hdu[0].data = all_imgs[i]
            # new_header = fix_NIRES_WCS(hdu[0].header)
            # new_header['HISTORY'] = 'Flat field corrected'
            # hdu[0].header = new_header
            hdu[0].header = all_heads[i]
            hdu[0].header['HISTORY'] = 'Flat field corrected'
            # hdul = fits.HDUList([hdu])
            hdu.writeto(out_name, overwrite=True)

    ########################################BACKGROUND SUB#####################################################

    # To do. Implement a two pass system where sources are masked.
    fns = []
    # all_dist = []
    # all_pdist = []
    # all_dist_id = []
    # all_sources = []
    targets_to_process = [x for x in np.unique(all_targets) if
                          not x.startswith('HIP')]  # Can change this to only process some targets

    if verbose:
        print('The following targets are found in the headers: ')
        print(np.unique(all_targets))

        print('The following targets are getting processed: ')
        print(targets_to_process)

    targets_to_process = ['2023fyq_S1']

    for target in np.unique(all_targets):
        if target not in targets_to_process: # or '_S1' in target or '202' not in target:
            continue  # as in quit the loop and continue
        print("Processing ", target)
        target_inds = all_targets == target
        if f'{target}_S1' in np.unique(all_targets):
            extra_target_inds = all_targets == f'{target}_S1'
            target_inds = target_inds + extra_target_inds
        start, stop = np.where(target_inds)[0][0], np.where(target_inds)[0][-1]
        # Just for fyq!!!---------------
        if target == '2023fyq' and directory == '20230729':
            start = start + 6
            target_inds[:7] = False
        target_imgs = all_imgs[target_inds, ...]
        # ---
        target_imgs = np.r_[target_imgs[12:], target_imgs[:12]]
        # ---
        # for i in range(target_imgs.shape[0]):
        #     mean, med, std = sigma_clipped_stats(target_imgs[i])
        #     plt.figure()
        #     plt.imshow(target_imgs[i], vmin=mean - 3 * std, vmax=mean + 3 * std)
        # plt.show()
        # return
        # target_imgs = target_imgs[2:, ...]
        target_heads = all_heads[start: stop + 1]
        # ---
        target_heads = target_heads[12:] + target_heads[:12]
        # ---
        targets = all_targets[target_inds]
        bad = input('Are there any bad images you want to drop?: ')
        bad = np.array(bad.split()).astype(int)
        if len(bad) > 0:
            bad = np.sort(bad)[::-1]
            target_imgs = np.delete(target_imgs, bad, axis=0)
            targets = np.delete(targets, bad, axis=0)
            for ind in bad:
                del target_heads[ind]
        S1_targets = np.array(['_S2' in targ for targ in targets])
        fns = file_list[target_inds]
        # If there's only one file, skip
        if target_imgs.shape[0] < 2:
            continue
        offsets = np.zeros(len(target_heads))
        # Figure out which images have the source outside the slit
        mask = np.ones((1024, 1024), dtype=np.uint8)
        mask[:50, :] = 0
        mask[-50:, :] = 0
        mask[520:, :40] = 0
        mask[800:, :80] = 0
        mask[900:, :120] = 0
        mask[940:, :175] = 0

        slit_centroid = 121.5, 465.3
        for i, head in enumerate(target_heads):
            offsets[i] = head['XOFFSET'] + head['YOFFSET']

        targ_coord = SkyCoord(f'{target_heads[0]["TARGRA"]} {target_heads[0]["TARGDEC"]}', unit=(u.deg, u.deg))
        targ_coord = SkyCoord("186.441125 12.663525", unit=(u.deg, u.deg))

        phot_imgs = target_imgs[(offsets == 0) & ~S1_targets]  # Only use images where source is not on slit for photometry
        # phot_imgs = target_imgs[:2, ...]

        # Measure median pixel value across image for background scale
        background_scales = []
        for img in target_imgs:
            #     for img in phot_imgs:
            background_scale = np.nanmedian(img.flatten())
            background_scales.append(background_scale)
        background_scales = np.array(background_scales)
        #     ratios = background_scales / np.nanmedian(background_scales)
        ###########################CHOICE TO COMBINE TO GET BKG IMAGE
        bg = np.median(phot_imgs, axis=0)
        #     print(len(phot_imgs))
        #     _,bg,_ = sigma_clipped_stats(phot_imgs, axis = 0, sigma = 5)
        #     _,bg,_ = sigma_clipped_stats(all_imgs, axis = 0, sigma = 5)
        #     bg = np.min(all_imgs,axis = 0)
        ratios = background_scales / np.nanmedian(bg)
        #     print(ratios.shape)
        # bg = np.nanmedian(astropy.stats.sigma_clip(target_imgs, axis=None, sigma_upper=3, sigma_lower=10), axis=0)

        # Write the backgrond frame is requested
        if write_bkg:
            hdu = fits.open(fns[0])
            hdu[0].data = bg
            hdu[0].header['HISTORY'] = 'Background frame'  # write more info here
            hdu.writeto(f'data/{directory}/redux/background/{target}_bkg.fits', overwrite=True)
        bg = bg[None, ...] * ratios[:, None, None]
        target_imgs -= bg

        # Write the background subtracted science image if requested
        if write_bkg_sub:
            for i, sub in enumerate(target_imgs):
                hdu = fits.open(fns[i])
                hdu[0].data = sub
                new_header = fix_NIRES_WCS(hdu[0].header)
                new_header['HISTORY'] = 'Background subtracted'  # write more info here
                hdu[0].header['HISTORY'] = new_header  # write more info here
                outname = fns[i].split('/')[-1]
                hdu.writeto(f'data/{directory}/redux/bkg_sub_{outname}', overwrite=True)

        # #############combine image###############

        S1_targets = np.array([False, False, False, False, False, False, False, False, False,
                               True, True, True, True, True, True, True, True])

        if S1_targets.sum() > 0:  # Need additional WCS care in this case
            ref_image = target_imgs[0, ...]
            slit_image = target_imgs[S1_targets != 0, ...][0, ...]
            mean, med, std = sigma_clipped_stats(slit_image)
            ref_image[ref_image < 0] = 0
            slit_image[slit_image < 0] = 0
            daofind = DAOStarFinder(fwhm=10, threshold=5 * std)
            sources = daofind(ref_image, mask=bpm).to_pandas().sort_values(by='flux', ascending=False)
            # Ignore sources too close to edge
            sources = sources[(sources['xcentroid'] > 50) & (sources['xcentroid'] < 974) &
                              (sources['ycentroid'] > 50) & (sources['ycentroid'] < 974)]
            sources = sources.reset_index(drop=True)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(ref_image, vmin=mean - 3 * std, vmax=mean + 3 * std)
            ax[1].imshow(slit_image, vmin=mean - 3 * std, vmax=mean + 3 * std)
            for i, row in sources.head(20).iterrows():
                source_aperture = CircularAperture((row.xcentroid, row.ycentroid), r=10)
                source_aperture.plot(color='w', lw=4, ax=ax[0])
                ax[0].annotate(i, (row.xcentroid, row.ycentroid))
            plt.show()

            ref_id = int(input('Reference source for correcting WCS offset: '))
            ref_coords = input('Rough coordinates of reference source in second image: ')
            ref_coords = np.array(ref_coords.split()).astype(int)
            ref_x1, ref_y1 = int(sources.iloc[ref_id].xcentroid), int(sources.iloc[ref_id].ycentroid)
            ref_x2, ref_y2 = ref_coords
            corr_size = 50
            xshift, yshift = image_registration.cross_correlation_shifts(
                ref_image[ref_y1 - corr_size: ref_y1 + corr_size, ref_x1 - corr_size: ref_x1 + corr_size],
                slit_image[ref_y2 - corr_size: ref_y2 + corr_size, ref_x2 - corr_size: ref_x2 + corr_size])
            xshift, yshift = ref_x2 - ref_x1 + xshift, ref_y2 - ref_y1 + yshift
            # plt.imshow(ref_image, vmin=mean + 2 * std, vmax=mean + 6 * std)
            # plt.figure()
            # plt.imshow(slit_image, vmin=mean + 2 * std, vmax=mean + 6 * std)
            # plt.figure()
            # plt.imshow(ref_image[ref_y1 - corr_size: ref_y1 + corr_size, ref_x1 - corr_size: ref_x1 + corr_size], vmin=0, vmax=100)
            # plt.figure()
            # plt.imshow(slit_image[ref_y2 - corr_size: ref_y2 + corr_size, ref_x2 - corr_size: ref_x2 + corr_size], vmin=0, vmax=100)
            # plt.show()
            x_scale, y_scale = 1, 1
            cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
            for i in range(target_imgs.shape[0]):
                if S1_targets[i]:
                    continue
                # plt.imshow(target_imgs[i], vmin=mean - 3 * std, vmax=mean + 3 * std)
                # wcs = WCS(target_heads[i])
                # target_pix = wcs.world_to_pixel(targ_coord)
                # target_aperture = CircularAperture(target_pix, r=10)
                # target_aperture.plot(lw=4, color='b')
                target_heads[i]['CRVAL1'] = target_heads[i]['CRVAL1'] + (
                        xshift * target_heads[i]['CD1_1'] * x_scale + yshift * target_heads[i][
                    'CD1_2'] * y_scale) / cosdec
                target_heads[i]['CRVAL2'] = target_heads[i]['CRVAL2'] + xshift * target_heads[i][
                    'CD2_1'] * x_scale + yshift * target_heads[0]['CD2_2'] * y_scale
                # wcs = WCS(target_heads[i])
                # target_pix = wcs.world_to_pixel(targ_coord)
                # target_aperture = CircularAperture(target_pix, r=10)
                # target_aperture.plot(lw=4, color='r')
                # plt.show()

        # Use the CORRECTED WCS to align. This should be good enough for now.
        for i in range(target_imgs.shape[0]):  # range(phot_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[0]))
            driz.add_image(target_imgs[i], WCS(target_heads[i]))
            target_imgs[i] = driz.outsci

        # for i in range(target_imgs.shape[0]):
        #     mean, med, std = sigma_clipped_stats(target_imgs[i])
        #     plt.figure()
        #     plt.imshow(target_imgs[i], vmin=mean - 3 * std, vmax=mean + 3 * std)
        # plt.show()

        orig_target_heads = deepcopy(target_heads)

        # ##########FINE ALIGNMENT FROM OLD SCRIPT###################

        if alignment_method == 'image_registration':
            # Align images using image_registration package around bright source
            align_images = target_imgs.copy()
            align_images[align_images < 0] = 0

            mean, median, std = sigma_clipped_stats(target_imgs[0][~bpm])
            # use bpm here
            daofind = DAOStarFinder(fwhm=10, threshold=5 * std)
            sources = daofind(target_imgs[0], mask=bpm).to_pandas().sort_values(by='flux', ascending=False)
            # Ignore sources too close to edge
            sources = sources[(sources['xcentroid'] > 50) & (sources['xcentroid'] < 974) &
                              (sources['ycentroid'] > 50) & (sources['ycentroid'] < 974)]
            sources = sources.reset_index(drop=True)
            plt.imshow(align_images[0], vmin=mean - 3 * std, vmax=mean + 3 * std)
            for i, row in sources.head(20).iterrows():
                source_aperture = CircularAperture((row.xcentroid, row.ycentroid), r=10)
                source_aperture.plot(color='w', lw=4)
                plt.annotate(i, (row.xcentroid, row.ycentroid))
            plt.show()
            ref_id = int(input('Reference source for image alignment: '))
            ref_x, ref_y = int(sources.iloc[ref_id].xcentroid), int(sources.iloc[ref_id].ycentroid)
            # ref_x, ref_y = int(sources.iloc[1].xcentroid), int(sources.iloc[1].ycentroid)
            corr_size = 100
            for i in range(target_imgs.shape[0]):
                xshift, yshift = image_registration.cross_correlation_shifts(
                    align_images[0][ref_y - corr_size: ref_y + corr_size, ref_x - corr_size: ref_x + corr_size],
                    align_images[i][ref_y - corr_size: ref_y + corr_size, ref_x - corr_size: ref_x + corr_size])
                # print(xshift, yshift)
                # plt.imshow(align_images[0], vmin=mean + 2 * std, vmax=mean + 6 * std)
                # plt.figure()
                # plt.imshow(align_images[i], vmin=mean + 2 * std, vmax=mean + 6 * std)
                # plt.figure()
                # plt.imshow(align_images[0][ref_y - corr_size: ref_y + corr_size, ref_x - corr_size: ref_x + corr_size], vmin=0, vmax=100)
                # plt.figure()
                # plt.imshow(align_images[i][ref_y - corr_size: ref_y + corr_size, ref_x - corr_size: ref_x + corr_size], vmin=0, vmax=100)
                # plt.show()
                print(i, xshift, yshift)
                x_scale, y_scale = -1, -1
                cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
                target_heads[i]['CRVAL1'] = target_heads[0]['CRVAL1'] + (
                        xshift * target_heads[0]['CD1_1'] * x_scale + yshift * target_heads[0][
                    'CD1_2'] * y_scale) / cosdec
                target_heads[i]['CRVAL2'] = target_heads[0]['CRVAL2'] + xshift * target_heads[0][
                    'CD2_1'] * x_scale + yshift * target_heads[0]['CD2_2'] * y_scale
                orig_target_heads[i]['CRVAL1'] = orig_target_heads[i]['CRVAL1'] + (
                        xshift * orig_target_heads[i]['CD1_1'] * x_scale + yshift * orig_target_heads[i][
                    'CD1_2'] * y_scale) / cosdec
                orig_target_heads[i]['CRVAL2'] = orig_target_heads[i]['CRVAL2'] + xshift * orig_target_heads[i][
                    'CD2_1'] * x_scale + yshift * orig_target_heads[i]['CD2_2'] * y_scale

                # print(xshift, yshift)
                # plt.figure()
                # plt.imshow(align_images[0], vmin=0, vmax=200)
                # plt.figure()
                # plt.imshow(align_images[i], vmin=0, vmax=200)
            # plt.show()
        elif alignment_method == 'cross_correlation':
            # Image shift using cross-correlations-------------------------------
            # Remove negative sources, just for cross-correlation
            align_images = target_imgs.copy()
            align_images[align_images < 0] = 0

            # This will work for an image containing only point sources, but will struggle with extended sources, need to
            # improve
            self_corr = scipy.signal.correlate(align_images[0], align_images[0], mode='same')
            ref1 = np.unravel_index(np.argmax(self_corr), self_corr.shape)
            for i in range(phot_imgs.shape[0]):
                cross_corr = scipy.signal.correlate(align_images[0][:412, 612:], align_images[i][:412, 612:],
                                                    mode='same')
                ref2 = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
                xshift, yshift = np.array(ref2) - np.array(ref1)
                x_scale, y_scale = 1, 1
                cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
                target_heads[i]['CRVAL1'] = target_heads[0]['CRVAL1'] + (
                        xshift * target_heads[0]['CD1_1'] * x_scale + yshift * target_heads[0][
                    'CD1_2'] * y_scale) / cosdec
                target_heads[i]['CRVAL2'] = target_heads[0]['CRVAL2'] + xshift * target_heads[0][
                    'CD2_1'] * x_scale + yshift * target_heads[0]['CD2_2'] * y_scale
        elif alignment_method == 'ref_source':
            # Image shift using reference source-----------------------------

            # Find reference sources in first image
            try:
                mean, median, std = sigma_clipped_stats(target_imgs[0][~bpm])

                # use bpm here
                daofind = DAOStarFinder(fwhm=10, threshold=5 * std)
                sources = daofind(target_imgs[0], mask=bpm).to_pandas().sort_values(by='flux', ascending=False)
                # Ignore sources too close to edge
                sources = sources[(sources['xcentroid'] > 50) & (sources['xcentroid'] < 974) &
                                  (sources['ycentroid'] > 50) & (sources['ycentroid'] < 974)]
                # Ignore very faint sources that are likely just noise
                ref_sources = sources[sources['mag'] < -2]
                max_source = ref_sources.iloc[0]
                N_sources = ref_sources.shape[0]
                # plt.figure()
                # positions = np.transpose((ref_sources['xcentroid'], ref_sources['ycentroid']))
                # apertures = CircularAperture(positions, r=8.0)
                # plt.imshow(target_imgs[0], cmap='gray', vmin=-50, vmax=500)
                # apertures.plot(color='blue', lw=2.5, alpha=0.5)
                # Find x/y shifts
                ref_lists = []
                for i in range(phot_imgs.shape[0]):
                    mean, median, std = sigma_clipped_stats(target_imgs[i][~bpm])
                    daofind = DAOStarFinder(fwhm=10, threshold=5 * std)
                    sources = daofind(target_imgs[i], mask=bpm).to_pandas().sort_values(by='flux', ascending=False)
                    search_rad = 30
                    ref_list = []
                    for j in range(N_sources):
                        xc, yc = ref_sources.iloc[j].xcentroid, ref_sources.iloc[j].ycentroid
                        ref_mag = ref_sources.iloc[j].mag
                        local_sources = sources.copy()
                        local_sources['sep'] = np.sqrt(
                            (local_sources.xcentroid - xc) ** 2 + (local_sources.ycentroid - yc) ** 2)
                        local_sources = local_sources.sort_values(by='sep')
                        positions = np.transpose((local_sources['xcentroid'], local_sources['ycentroid']))
                        apertures = CircularAperture(positions, r=8.0)
                        plt.imshow(target_imgs[0], cmap='gray', vmin=-50, vmax=500)
                        apertures.plot(color='blue', lw=2.5, alpha=0.5)
                        local_sources = local_sources[
                            local_sources.mag < ref_mag + 2]  # Don't select things much fainter than reference source
                        local_sources['sep'] = np.sqrt(
                            (local_sources.xcentroid - xc) ** 2 + (local_sources.ycentroid - yc) ** 2)
                        local_sources = local_sources[local_sources['sep'] < search_rad]
                        local_sources = local_sources.sort_values(by='sep')
                        if local_sources.empty:
                            ref_list.append([])
                        else:
                            ref_list.append([local_sources.iloc[0].xcentroid, local_sources.iloc[0].ycentroid])
                    ref_lists.append(ref_list)
                for i in range(phot_imgs.shape[0]):
                    ref_list = ref_lists[i]
                    x_shifts, y_shifts = [], []
                    for j in range(N_sources):
                        refs = ref_list[j]
                        if len(refs) == 0:
                            continue
                        x_shifts.append(ref_sources.iloc[j].xcentroid - refs[0])
                        y_shifts.append(ref_sources.iloc[j].ycentroid - refs[1])
                    x_shift, y_shift = np.median(x_shifts), np.median(y_shifts)
                    print(x_shifts, y_shifts)
                    x_scale, y_scale = 1, 1
                    cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
                    target_heads[i]['CRVAL1'] = target_heads[0]['CRVAL1'] + (
                            x_shift * target_heads[0]['CD1_1'] * x_scale + y_shift * target_heads[0][
                        'CD1_2'] * y_scale) / cosdec
                    target_heads[i]['CRVAL2'] = target_heads[0]['CRVAL2'] + x_shift * target_heads[0][
                        'CD2_1'] * x_scale + y_shift * target_heads[0]['CD2_2'] * y_scale
                    # Save intermediate images
            except:
                print("Fine alignment fails. Probably no sources found! Skipping for now")
                continue

        seps, pas = [], []
        # Correct for astrometry based on slit position
        for i, head in enumerate(target_heads):
            if offsets[i] != 0:
                wcs = WCS(orig_target_heads[i])
                slit_coord = wcs.pixel_to_world(*slit_centroid)
                ref_coord = slit_coord.directional_offset_by(head['ROTDEST'] * u.deg, (offsets[i] / 3600) * u.deg)
                sep, pa = ref_coord.separation(targ_coord).to(u.deg), ref_coord.position_angle(targ_coord).to(u.deg)
                seps.append(sep.value)
                pas.append(pa.value)
                # mean, med, std = sigma_clipped_stats(target_imgs[i])
                # wcs = WCS(target_heads[0])
                # plt.figure()
                # plt.imshow(target_imgs[0], vmin=mean + 3 * std, vmax=mean + 6 * std)
                # ref_pix = wcs.world_to_pixel(ref_coord)
                # for plot_sep in np.linspace(0, sep, 10):
                #     plot_coord = targ_coord.directional_offset_by(pa, -plot_sep)
                #     plot_pix = wcs.world_to_pixel(plot_coord)
                #     plot_aperture = CircularAperture(plot_pix, r=4)
                #     plot_aperture.plot(color='green', lw=2)
                # target_aperture = CircularAperture(wcs.world_to_pixel(slit_coord), r=10)
                # target_aperture.plot(color='blue', lw=4)
                # target_aperture = CircularAperture(wcs.world_to_pixel(targ_coord), r=10)
                # target_aperture.plot(color='k', lw=4)
                # target_aperture = CircularAperture(ref_pix, r=10)
                # target_aperture.plot(color='red', lw=4)
                # plt.show()

                # plt.figure()
                # ax = plt.subplot(projection=wcs)
                # mean, med, std = sigma_clipped_stats(target_imgs[i])
                # ax.imshow(target_imgs[i], vmin=mean - 3 * std, vmax=mean + 3 * std)
                # ref_pix = wcs.world_to_pixel(ref_coord)
                # ref_aperture = CircularAperture(ref_pix, r=10)
                # target_pix = wcs.world_to_pixel(targ_coord)
                # target_aperture = CircularAperture(target_pix, r=10)
                # ref_aperture.plot(color='red', lw=4)
                # target_aperture.plot(color='blue', lw=4)
                # for plot_sep in np.linspace(0, sep, 10):
                #     print(plot_sep, sep)
                #     plot_coord = targ_coord.directional_offset_by(pa, -plot_sep)
                #     plot_pix = wcs.world_to_pixel(plot_coord)
                #     plot_aperture = CircularAperture(plot_pix, r=4)
                #     plot_aperture.plot(color='green', lw=2)
                #     print(plot_pix)
                # plt.show()

        # print(seps)
        # print(pas)
        sep, pa = np.median(seps), np.median(pas)
        # sep = sep * 0.86
        # pa = pa - 23.5  # 4.3

        for i in range(len(target_heads)):
            head = target_heads[i]
            init_coord = SkyCoord(head['CRVAL1'], head['CRVAL2'], unit=(u.deg, u.deg))
            updated_coord = init_coord.directional_offset_by(pa * u.deg, sep * u.deg)
            target_heads[i]['CRVAL1'] = updated_coord.ra.value
            target_heads[i]['CRVAL2'] = updated_coord.dec.value

            # wcs = WCS(target_heads[i])
            # plt.figure()
            # ax = plt.subplot(projection=wcs)
            # mean, med, std = sigma_clipped_stats(target_imgs[i])
            # ax.imshow(target_imgs[i], vmin=mean + 2 * std, vmax=mean + 8 * std)
            # target_pix = wcs.world_to_pixel(targ_coord)
            # target_aperture = CircularAperture(target_pix, r=10)
            # target_aperture.plot(color='blue', lw=4)
            # print(target_pix)
            # for plot_sep in np.linspace(0, sep, 10):
            #     plot_coord = targ_coord.directional_offset_by(pa * u.deg, -plot_sep * u.deg)
            #     plot_pix = wcs.world_to_pixel(plot_coord)
            #     plot_aperture = CircularAperture(plot_pix, r=4)
            #     plot_aperture.plot(color='green', lw=2)
            #     print(plot_pix)
            # plt.show()
            # continue
            #
            # mean, med, std = sigma_clipped_stats(target_imgs[i])
            # plt.figure()
            # ax = plt.subplot(projection=wcs)
            # ax.imshow(target_imgs[i], vmin=mean + 4 * std, vmax=mean + 10 * std)
            # target_pix = wcs.world_to_pixel(targ_coord)
            # target_aperture = CircularAperture(target_pix, r=10)
            # target_aperture.plot(color='red', lw=4)
            # # plt.show()

        for i in range(phot_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[i]))
            driz.add_image(target_imgs[i], WCS(target_heads[i]))
            driz.write(f'data/{directory}/redux/{target}_{i}.fits.gz')
        # Combine images
        wcs = WCS(target_heads[0])
        driz = Drizzle(outwcs=wcs)
        for i in range(phot_imgs.shape[0]):
            driz.add_image(target_imgs[i], WCS(target_heads[i]))

        # Alignment to ensure that target RA/Dec lie exactly on source position
        mean, med, std = sigma_clipped_stats(driz.outsci)
        daofind = DAOStarFinder(fwhm=5, threshold=5 * std)
        sources = daofind(driz.outsci - med).to_pandas()
        target_pix = wcs.world_to_pixel(targ_coord)

        # Fine adjustment of WCS
        sources['sep'] = np.sqrt(
            (target_pix[0] - sources['xcentroid']) ** 2 + (target_pix[1] - sources['ycentroid']) ** 2)
        max_source = sources[sources.sep == sources.sep.min()]

        xshift, yshift = target_pix[0] - max_source.xcentroid.values[0], target_pix[1] - max_source.ycentroid.values[0]
        x_scale, y_scale = 1, 1
        cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
        for i in range(phot_imgs.shape[0]):
            target_heads[i]['CRVAL1'] = target_heads[i]['CRVAL1'] + (
                    xshift * target_heads[i]['CD1_1'] * x_scale + yshift * target_heads[i][
                'CD1_2'] * y_scale) / cosdec
            target_heads[i]['CRVAL2'] = target_heads[i]['CRVAL2'] + xshift * target_heads[i][
                'CD2_1'] * x_scale + yshift * target_heads[0]['CD2_2'] * y_scale
        wcs = WCS(target_heads[0])

        driz = Drizzle(outwcs=wcs)
        for i in range(phot_imgs.shape[0]):
            driz.add_image(target_imgs[i], WCS(target_heads[i]))

        driz.write(f'data/{directory}/redux/combined/{target}.fits.gz')

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(driz.outsci, origin='lower', vmin=mean-8*std, vmax=mean+90*std)
        plt.xlim(target_pix[0] - 100, target_pix[0] + 100)
        plt.ylim(target_pix[1] - 100, target_pix[1] + 100)
        target_pix = wcs.world_to_pixel(targ_coord)
        target_aperture = CircularAperture(target_pix, r=10)
        ap_patches = target_aperture.plot(color='blue', lw=2)

        plt.show()

        # offset = input('At this point, the aperture should lie right on the target. Do you want to apply a pixel offset?: ')
        # if offset.lower() in ['y', 'yes']:
        #     pix_offset = input('What pixel offset would you like to apply?')
        #     pix_offset = np.array(pix_offset.split(' '))
        #     print(pix_offset)


        image = fits.open(f'data/{directory}/redux/combined/{target}.fits.gz')

        # print("Querying Vizier for 2MASS and UKIDSS stars.")
        # std_cat = query_cat(targ_coord, size=10, catalog=None)
        # print("Running source finding and catalog matching.")
        # sources, matches = find_and_match_sources(image[1].data, image[1].header, std_cat, distance_pix=30, plot=True)
        # print("Performing photometry")
        # radii = [1, 3, 5, 7, 9, 11, 15]
        # ap_photo = apPhot(image[1].data, image[1].data, sources, radii=radii)
        # print("Calculating zeropoint")
        # # pdb.set_trace()
        # final_ZP, final_ZP_err = ZPCalculation('K', sources, std_cat, matches, ap_photo, radii, plot=True)
        #
        # SN_pix = wcs.world_to_pixel(targ_coord) + np.array([-3, 0])
        # SN_source = Table([[SN_pix[0]], [SN_pix[1]]], names=['xcentroid', 'ycentroid'])
        #
        # ########################################################
        # fig = plt.figure(figsize=(10, 10))
        # ax = plt.subplot(projection=wcs)
        #
        # # THESE ARE JUST FOR PLOTTING
        # aperture = CircularAperture(SN_pix, r=7)
        # annulus = CircularAnnulus(SN_pix, r_in=7 * 1.5, r_out=7 * 3)
        #
        # ax.imshow(image[1].data, origin='lower', vmin=mean - 5 * std, vmax=mean + 70 * std)
        #
        # ax.set_xlim([SN_pix[0] - 100, SN_pix[0] + 100])
        # ax.set_ylim([SN_pix[1] - 100, SN_pix[1] + 100])
        #
        # ap_patches = aperture.plot(color='white', lw=2)
        # an_patches = annulus.plot(color='red', lw=2)
        # plt.tight_layout()
        # plt.show()
        # ########################################################
        #
        # # SN_phot = apPhot(image[0].data, image[1].data,SN_source, radii = radii)
        #
        # # pdb.set_trace()
        #
        # SN_phot = apPhot(image[1].data, image[1].data, sources[[int(x) for x in matches[:, 0]]], radii=radii)
        #
        # # print(SN_phot)
        #
        # mag, emag, limit_3sig_SN, limit_5sig_SN = computeMag(SN_phot, final_ZP, final_ZP_err)
        #
        # # print(mag)
        # # print(emag)
        # outTable = Table([radii, mag, emag, limit_3sig_SN, limit_5sig_SN],
        #                  names=['radii_pix', 'mag', 'emag', 'mag_limit_3sig', 'mag_limit_5sig'])
        # print(outTable)
        #
        # continue

        Vizier.ROW_LIMIT = -1
        result = Vizier.query_region(targ_coord, radius=0.05 * u.deg, catalog=["II/246/out", "II/319/las9"])
        good_tables = []
        for i, table in enumerate(result):
            if table.meta['ID'] == 'II_246_out':
                table = table[(table['Qflg'] == 'AAA') & (table[0]['Bflg'] == '111')]
            elif table.meta['ID'] == 'II_319_las9':
                table = table[table['Kmag'] < 20]
            table = table['RAJ2000', 'DEJ2000']
            good_tables.append(table)
        good_sources = vstack([*good_tables])

        # good2M = result[0][(result[0]['Qflg'] == 'AAA') & (result[0]['Bflg'] == '111')]
        # good_UKIDSS = result[1][result[1]['Jmag1'] < 19]
        mean, med, std = sigma_clipped_stats(stacked_img)

        daofind = DAOStarFinder(fwhm=5, threshold=5 * std, sharphi=0.55)
        sources = daofind(stacked_img - med)
        sources = sources[sources['flux'] > 1.5]
        ref_coord = SkyCoord(ra=good_sources['RAJ2000'], dec=good_sources['DEJ2000'], unit=(u.deg, u.deg))
        ref_pix = wcs.world_to_pixel(ref_coord)

        target_pix = wcs.world_to_pixel(targ_coord)
        target_aperture = CircularAperture(target_pix, r=10)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot()  # projection=wcs
        ax.imshow(stacked_img, origin='lower', vmin=median - 3 * std, vmax=median + 3 * std)
        ap_patches = target_aperture.plot(color='blue', lw=4)

        for i, row in sources.to_pandas().iterrows():
            source_aperture = CircularAperture((row.xcentroid, row.ycentroid), r=10)
            source_aperture.plot(color='w', lw=4)

        matches_px = []
        for ind, i in enumerate(ref_pix[0]):
            x, y = (ref_pix[0][ind], ref_pix[1][ind])
            matches_px += [[x, y]]
        if len(matches_px) == 0:
            print(f'Skipping {target}, no reference sources found in catalogue to build WCS')
            continue

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot()  # projection=wcs
        ax.imshow(stacked_img, origin='lower', vmin=median - 3 * std, vmax=median + 3 * std)

        ap_matches = CircularAperture(matches_px, r=10)
        foo = ap_matches.plot(color='r', lw=1)

        plt.show()

        # Match sources found to UKIDSS
        matches = []
        for ind, i in enumerate(ref_pix[0]):
            if ref_pix[0][ind] < 0 or ref_pix[0][ind] > 1024 or ref_pix[1][ind] < 0 or ref_pix[1][ind] > 1024:
                continue
            dist = np.sqrt((np.array(sources['xcentroid']) - ref_pix[0][ind]) ** 2 +
                           (np.array(sources['ycentroid']) - ref_pix[1][ind]) ** 2)
            idx = np.argmin(dist)
            if np.min(dist) < 50:
                matches += [[idx, ind, np.min(dist)]]
        if len(matches) == 0:
            print(f'Skipping {target}, no reference sources found in catalogue to build WCS')
            continue

        matches = np.array(matches)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot()  # projection=wcs

        ax.imshow(stacked_img, origin='lower', vmin=mean - 3 * std, vmax=mean + 3 * std)

        target_pix = wcs.world_to_pixel(targ_coord)
        target_aperture = CircularAperture(target_pix, r=10)

        ap_patches = target_aperture.plot(color='blue', lw=2)

        matches_px = []
        for ind, i in enumerate(matches):
            x, y = sources[int(i[0])]['xcentroid'], sources[int(i[0])]['ycentroid']
            matches_px += [[x, y]]

        ap_matches = CircularAperture(matches_px, r=10)
        foo = ap_matches.plot(color='white', lw=1)

        ###########FIT FOR A NEW WCS
        new_wcs = fit_wcs_from_points((sources[matches[:, 0].astype('int')]['xcentroid'],
                                       sources[matches[:, 0].astype('int')]['ycentroid']),
                                      ref_coord[matches[:, 1].astype('int')], proj_point='center', projection='TAN',
                                      sip_degree=1)
        new_header = new_wcs.to_header(relax=True)
        wcs = new_wcs

        for ext in stack:
            for j in new_header:
                ext.header[j] = new_header[j]

        stack.writeto(f'data/{directory}/redux/fixed_astrometry/{target}.fits.gz', overwrite=True)
        # except:
        #     print('Error doing astrometry, continuing to next source')
        #     plt.close('all')
        #     continue

        final_im = fits.open(f'data/{directory}/redux/fixed_astrometry/{target}.fits.gz')

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(projection=new_wcs)

        ax.imshow(final_im[1].data, origin='lower', vmin=mean - 3 * std, vmax=mean + 8 * std)

        # positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')
        SN_pix = new_wcs.world_to_pixel(targ_coord)
        aperture = CircularAperture(SN_pix, r=10)

        size = 500
        ax.set_xlim([SN_pix[0] - size, SN_pix[0] + size])
        ax.set_ylim([SN_pix[1] - size, SN_pix[1] + size])

        ap_patches = aperture.plot(color='blue', lw=2)

        #########PLOT UKIDSS to check
        ap_UKIDSS = []
        for i in good_sources:
            coord = SkyCoord(ra=i['RAJ2000'] * u.deg, dec=i['DEJ2000'] * u.deg)
            pix = new_wcs.world_to_pixel(coord)
            fake_ap = CircularAperture(pix, r=10)
            ap_stats = ApertureStats(final_im[1].data, fake_ap)
            #     print(pix, ap_stats.centroid)
            #     print(ap_stats.centroid[0])
            if ~np.isnan(ap_stats.centroid[0]):
                ap_UKIDSS += [ap_stats.centroid]
        #     bad_UKIDSS += [pix]
        #     ap_UKIDSS += [pix]

        aperture_UKIDSS = CircularAperture(ap_UKIDSS, r=10)
        # foo = photutils.CircularAperture(bad_UKIDSS, r=10)
        # annulus_UKIDSS = CircularAnnulus(ap_UKIDSS, r_in=15, r_out = 20)

        foo = aperture_UKIDSS.plot(color='red', lw=1)
        # foo2 = annulus_UKIDSS.plot(color = 'red', lw = 1)

        # Do photometry-------------------------------
        result = Vizier.query_region(targ_coord, width="3m", catalog=["II/246/out", "II/319/las9"])
        good_tables = []
        for i, table in enumerate(result):
            if table.meta['ID'] == 'II_246_out':
                continue
                table = table[(table['Qflg'] == 'AAA') & (table[0]['Bflg'] == '111')]
            elif table.meta['ID'] == 'II_319_las9':
                table = table[table['Kmag'] < 18]
            table = table['RAJ2000', 'DEJ2000', 'Kmag', 'e_Kmag']
            good_tables.append(table)
        good_sources = vstack([*good_tables])
        # good2M = result[0][(result[0]['Qflg'] == 'AAA') & (result[0]['Bflg'] == '111')]
        # good_UKIDSS = result[1][result[1]['Kmag'] < 18]

        # positions = SkyCoord(catalog['l'], catalog['b'], frame='galactic')
        target_pix = wcs.world_to_pixel(targ_coord)

        aperture_target = CircularAperture(target_pix, r=10)
        ap_stats = ApertureStats(final_im[1].data, aperture_target)
        aperture_target = CircularAperture(ap_stats.centroid, r=10)
        annulus_target = CircularAnnulus(ap_stats.centroid, r_in=15, r_out=20)

        ap_ref = []
        ref_good = []
        for ind, i in enumerate(good_sources):
            coord = SkyCoord(ra=i['RAJ2000'] * u.deg, dec=i['DEJ2000'] * u.deg)
            pix = wcs.world_to_pixel(coord)
            if pix[0] < 50 or pix[0] > 974 or pix[1] < 50 or pix[1] > 974:  # Skip sources too close to edge
                continue
            ref_good.append(ind)
            fake_ap = CircularAperture(pix, r=10)
            ap_stats = ApertureStats(final_im[1].data, fake_ap)
            if ~np.isnan(ap_stats.centroid[0]):
                ap_ref += [ap_stats.centroid]
            else:
                ap_ref += [pix]
        ref_good = np.array(ref_good)

        aperture_ref = CircularAperture(ap_ref, r=5)
        annulus_ref = CircularAnnulus(ap_ref, r_in=10, r_out=15)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(projection=wcs)
        ax.imshow(stacked_img, origin='lower', vmin=mean - 3 * std, vmax=mean + 8 * std)
        ap_patches = aperture_target.plot(color='blue', lw=2)
        ap_patches = annulus_target.plot(color='red', lw=2)

        aperture_ref.plot(color='white', lw=1)
        annulus_ref.plot(color='red', lw=1)
        for i in range(len(aperture_ref)):
            ax.text(aperture_ref.positions[i][0], aperture_ref.positions[i][1], i, fontsize=18)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(projection=wcs)
        ax.imshow(stacked_img, origin='lower', vmin=mean-8*std, vmax=mean+90*std)
        ap_patches = aperture_target.plot(color='blue', lw=2)
        ap_patches = annulus_target.plot(color='red', lw=2)

        aperture_ref.plot(color='white', lw=1)
        annulus_ref.plot(color='red', lw=1)
        for i in range(len(aperture_ref)):
            ax.text(aperture_ref.positions[i][0], aperture_ref.positions[i][1], i, fontsize=18)

        plt.show()

        good_ids = input('Which ref sources are good and should be used?: ')
        good_ids = np.array(good_ids.split()).astype(int)

        phot_table_target = aperture_photometry(final_im[1].data, aperture_target,
                                                          error=np.sqrt(final_im[2].data))
        phot_bkg_target = aperture_photometry(final_im[1].data, annulus_target,
                                                        error=np.sqrt(final_im[2].data))

        target_bkg_sub = phot_table_target[0]['aperture_sum'] - phot_bkg_target[0]['aperture_sum'] \
                         / annulus_target.area * aperture_target.area
        target_bkg_sub_err = np.sqrt(phot_table_target[0]['aperture_sum_err'] ** 2 +
                                     (phot_bkg_target[0]['aperture_sum_err'] /
                                      annulus_target.area * aperture_target.area) ** 2)

        phot_table_UKIDSS = aperture_photometry(final_im[1].data, aperture_ref, error=final_im[2].data)
        phot_bkg_UKIDSS = aperture_photometry(final_im[1].data, annulus_ref, error=final_im[2].data)

        phot_table_UKIDSS = phot_table_UKIDSS[good_ids]
        phot_bkg_UKIDSS = phot_bkg_UKIDSS[good_ids]

        UKIDSS_bkg_sub = phot_table_UKIDSS['aperture_sum'] - phot_bkg_UKIDSS['aperture_sum'] / \
                         annulus_ref.area * aperture_ref.area

        UKIDSS_bkg_sub_err = np.sqrt(phot_table_UKIDSS['aperture_sum_err'] ** 2 +
                                     (phot_bkg_UKIDSS[
                                          'aperture_sum_err'] / annulus_ref.area * aperture_ref.area) ** 2)

        mags = good_sources[good_ids]['Kmag']
        mag_err = good_sources[good_ids]['e_Kmag']

        ZP = mags + 2.5 * np.log10(UKIDSS_bkg_sub)
        ZP_err = np.sqrt(mag_err ** 2 + (2.5 / UKIDSS_bkg_sub / np.log(10) * UKIDSS_bkg_sub_err) ** 2)

        target_mag = -2.5 * np.log10(target_bkg_sub) + ZP
        target_emag = np.sqrt(ZP_err ** 2 + (2.5 / target_bkg_sub / np.log(10) * target_bkg_sub_err) ** 2)

        target_mag, target_emag = target_mag[~np.isnan(target_mag)], target_emag[~np.isnan(target_mag)]

        av_target_mag = np.average(target_mag, weights=1 / target_emag)
        av_target_mag_err = np.sqrt(np.cov(target_mag, aweights=1 / target_emag))

        print("K = %f +- %f mag" % (av_target_mag, av_target_mag_err))

        # driz.write(f'data/{directory}/redux/combined/{target}.fits.gz')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection=new_wcs)
        # plt.imshow(driz.outsci, vmin=0, vmax=100)
        # plt.scatter(targ_coord.ra, targ_coord.dec, transform=ax.get_transform('world'), facecolors='none',
        #             edgecolors='r')
        # plt.show()


reduceNight('20240615', write_flat=False, write_flatten=False, write_bkg=False, write_bkg_sub=False)
