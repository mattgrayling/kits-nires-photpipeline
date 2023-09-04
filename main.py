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
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack
from astropy.stats import sigma_clip
from drizzle.drizzle import Drizzle
import photutils
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus
import astroalign as aa
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astroquery.vizier import Vizier

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
    PA = header['ROTDEST']  # PA of vertical axis of the image
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

    for i, file in enumerate(file_list):
        with fits.open(file) as hdu:
            head = hdu[0].header
            img = hdu[0].data
        target = head['TARGNAME']
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

    for target in np.unique(all_targets):
        # targets_to_process = ['2022crv']
        # targets_to_process = ['2022jli_S1']
        if target not in targets_to_process:
            continue  # as in quit the loop and continue
        # if target == '13232490+444059':
        #     continue
        print("Processing ", target)
        target_inds = all_targets == target
        # extra_target_inds = all_targets == f'{target}_S1'
        # target_inds = target_inds + extra_target_inds
        start, stop = np.where(target_inds)[0][0], np.where(target_inds)[0][-1]
        target_imgs = all_imgs[target_inds, ...]
        target_heads = all_heads[start: stop + 1]
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

        slit_centroids = np.zeros((target_imgs.shape[0], 2))

        for i, head in enumerate(target_heads):
            offsets[i] = head['XOFFSET'] + head['YOFFSET']
            # if offsets[i] > 0:
            test_image = target_imgs[i].copy()
            test_image[test_image > np.median(test_image) - 3 * np.std(test_image)] = 0
            test_image = test_image / test_image.max()
            test_image = (test_image > 0.01).astype(np.uint8)
            test_image_comp = cv2.bitwise_not(test_image)
            # Remove single pixels
            kernel1 = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]], np.uint8)
            kernel2 = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], np.uint8)
            hitormiss1 = cv2.morphologyEx(test_image, cv2.MORPH_ERODE, kernel1)
            hitormiss2 = cv2.morphologyEx(test_image_comp, cv2.MORPH_ERODE, kernel2)
            hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
            test_image -= hitormiss
            test_image *= mask
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(test_image)
            slit_ind = np.argsort(stats[:, -1])[-2]
            slit_centroid = (centroids[slit_ind, 0], centroids[slit_ind, 1])
            slit_centroids[i, ...] = slit_centroid

        slit_centroid = np.median(slit_centroids, axis=0)

        targ_coord = SkyCoord(f'{head["TARGRA"]} {head["TARGDEC"]}', unit=(u.deg, u.deg))

        seps, pas = [], []
        # Correct for astrometry based on slit position
        for i, head in enumerate(target_heads):
            if offsets[i] != 0:
                head = target_heads[i]
                wcs = WCS(head)
                slit_coord = wcs.pixel_to_world(*slit_centroid)
                ref_coord = slit_coord.directional_offset_by(head['PA'], (offsets[i] / 3600) * u.deg)
                sep, pa = ref_coord.separation(targ_coord).to(u.deg), ref_coord.position_angle(targ_coord).to(u.deg)
                seps.append(sep.value)
                pas.append(pa.value)
        sep, pa, = np.median(seps), np.median(pas)

        for i in range(len(target_heads)):
            head = target_heads[i]
            init_coord = SkyCoord(head['CRVAL1'], head['CRVAL2'], unit=(u.deg, u.deg))
            updated_coord = init_coord.directional_offset_by(pa * u.deg, sep * u.deg)
            target_heads[i]['CRVAL1'] = updated_coord.ra.value
            target_heads[i]['CRVAL2'] = updated_coord.dec.value

        phot_imgs = target_imgs[offsets == 0]  # Only use images where source is not on slit for photometry

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
        # Use the CORRECTED WCS to align. This should be good enough for now.
        for i in range(target_imgs.shape[0]):  # range(phot_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[0]))
            driz.add_image(target_imgs[i], WCS(target_heads[i]))
            target_imgs[i] = driz.outsci

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
            ref_x, ref_y = int(sources.iloc[0].xcentroid), int(sources.iloc[0].ycentroid)

            for i in range(phot_imgs.shape[0]):
                xshift, yshift = image_registration.cross_correlation_shifts(
                    align_images[0][ref_y - 50: ref_y + 50, ref_x - 50: ref_x + 50],
                    align_images[i][ref_y - 50: ref_y + 50, ref_x - 50: ref_x + 50])
                x_scale, y_scale = -1, -1
                cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
                target_heads[i]['CRVAL1'] = target_heads[0]['CRVAL1'] + (
                        xshift * target_heads[0]['CD1_1'] * x_scale + yshift * target_heads[0][
                    'CD1_2'] * y_scale) / cosdec
                target_heads[i]['CRVAL2'] = target_heads[0]['CRVAL2'] + xshift * target_heads[0][
                    'CD2_1'] * x_scale + yshift * target_heads[0]['CD2_2'] * y_scale

            #     print(xshift, yshift)
            #     plt.figure()
            #     plt.imshow(align_images[0], vmin=0, vmax=200)
            #     plt.figure()
            #     plt.imshow(align_images[i], vmin=0, vmax=200)
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
                print(sources.shape)
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

        for i in range(phot_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[i]))
            driz.add_image(target_imgs[i], WCS(target_heads[i]))
            driz.write(f'data/{directory}/redux/{target}_{i}.fits.gz')
        # Combine images
        wcs = WCS(target_heads[0])
        driz = Drizzle(outwcs=wcs)
        for i in range(phot_imgs.shape[0]):
            driz.add_image(target_imgs[i], WCS(target_heads[i]))

        driz.write(f'data/{directory}/redux/combined/{target}_{i}.fits.gz')

        # Fine adjustment of WCS
        try:
            stack = fits.open(f'data/{directory}/redux/combined/{target}_{i}.fits.gz')
            stacked_img = stack[1].data
            # Fine adjustment of WCS
            Vizier.ROW_LIMIT = -1
            result = Vizier.query_region(targ_coord, width="6m", catalog=["II/246/out", "II/319/las9"])
            good_tables = []
            for i, table in enumerate(result):
                if table.meta['ID'] == 'II_246_out':
                    table = table#[(table['Qflg'] == 'AAA') & (table[0]['Bflg'] == '111')]
                elif table.meta['ID'] == 'II_319_las9':
                    table = table[table['Jmag1'] < 19]
                table = table['RAJ2000', 'DEJ2000']
                good_tables.append(table)
            good_sources = vstack([*good_tables])
            # good2M = result[0][(result[0]['Qflg'] == 'AAA') & (result[0]['Bflg'] == '111')]
            # good_UKIDSS = result[1][result[1]['Jmag1'] < 19]
            mean, med, std = sigma_clipped_stats(stacked_img)

            sources = daofind(stacked_img - med)
            ref_coord = SkyCoord(ra=good_sources['RAJ2000'], dec=good_sources['DEJ2000'], unit=(u.deg, u.deg))
            ref_pix = wcs.world_to_pixel(ref_coord)

            target_pix = wcs.world_to_pixel(targ_coord)
            aperture_ann = CircularAperture(target_pix, r=5)

            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(projection=wcs)
            ax.imshow(stacked_img, origin='lower', vmin=mean - 3 * std, vmax=mean + 5 * std)
            ap_patches = aperture_ann.plot(color='blue', lw=2)

            matches_px = []
            for ind, i in enumerate(ref_pix[0]):
                x, y = (ref_pix[0][ind], ref_pix[1][ind])
                matches_px += [[x, y]]

            ap_matches = CircularAperture(matches_px, r=10)
            foo = ap_matches.plot(color='white', lw=1)

            # Match sources found to UKIDSS
            matches = []
            for ind, i in enumerate(ref_pix[0]):
                dist = np.sqrt((np.array(sources['xcentroid']) - ref_pix[0][ind]) ** 2 +
                               (np.array(sources['ycentroid']) - ref_pix[1][ind]) ** 2)
                idx = np.argmin(dist)
                if np.min(dist) < 50:
                    matches += [[idx, ind, np.min(dist)]]
            matches = np.array(matches)

            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(projection=wcs)

            ax.imshow(stacked_img, origin='lower', vmin=mean - 3 * std, vmax=mean + 5 * std)

            target_pix = wcs.world_to_pixel(targ_coord)
            aperture_ann = CircularAperture(target_pix, r=5)

            ap_patches = aperture_ann.plot(color='blue', lw=2)

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
        except:
            print('Error doing astrometry, continuing to next source')
            plt.close('all')
            continue

        final_im = fits.open(f'data/{directory}/redux/fixed_astrometry/{target}.fits.gz')

        # Do photometry-------------------------------
        result = Vizier.query_region(targ_coord, width="3m", catalog=["II/246/out", "II/319/las9"])
        good_tables = []
        for i, table in enumerate(result):
            if table.meta['ID'] == 'II_246_out':
                table = table#[(table['Qflg'] == 'AAA') & (table[0]['Bflg'] == '111')]
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

        aperture_ref = CircularAperture(ap_ref, r=15)
        annulus_ref = CircularAnnulus(ap_ref, r_in=20, r_out=30)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(projection=wcs)
        ax.imshow(stacked_img, origin='lower', vmin=0, vmax=100)
        ap_patches = aperture_target.plot(color='blue', lw=2)
        ap_patches = annulus_target.plot(color='red', lw=2)

        aperture_ref.plot(color='white', lw=1)
        annulus_ref.plot(color='red', lw=1)
        for i in range(len(aperture_ref)):
            ax.text(aperture_ref.positions[i][0], aperture_ref.positions[i][1], i, fontsize=18)

        phot_table_target = photutils.aperture_photometry(final_im[1].data, aperture_target,
                                                          error=np.sqrt(final_im[2].data))
        phot_bkg_target = photutils.aperture_photometry(final_im[1].data, annulus_target,
                                                        error=np.sqrt(final_im[2].data))

        target_bkg_sub = phot_table_target[0]['aperture_sum'] - phot_bkg_target[0]['aperture_sum'] \
                         / annulus_target.area * aperture_target.area
        target_bkg_sub_err = np.sqrt(phot_table_target[0]['aperture_sum_err'] ** 2 +
                                     (phot_bkg_target[0]['aperture_sum_err'] /
                                      annulus_target.area * aperture_target.area) ** 2)

        phot_table_UKIDSS = photutils.aperture_photometry(final_im[1].data, aperture_ref, error=final_im[2].data)
        phot_bkg_UKIDSS = photutils.aperture_photometry(final_im[1].data, annulus_ref, error=final_im[2].data)

        UKIDSS_bkg_sub = phot_table_UKIDSS['aperture_sum'] - phot_bkg_UKIDSS['aperture_sum'] / \
                         annulus_ref.area * aperture_ref.area

        UKIDSS_bkg_sub_err = np.sqrt(phot_table_UKIDSS['aperture_sum_err'] ** 2 +
                                     (phot_bkg_UKIDSS[
                                          'aperture_sum_err'] / annulus_ref.area * aperture_ref.area) ** 2)

        mags = good_sources[ref_good]['Kmag']
        mag_err = good_sources[ref_good]['e_Kmag']

        ZP = mags + 2.5 * np.log10(UKIDSS_bkg_sub)
        ZP_err = np.sqrt(mag_err ** 2 + (2.5 / UKIDSS_bkg_sub / np.log(10) * UKIDSS_bkg_sub_err) ** 2)

        target_mag = -2.5 * np.log10(target_bkg_sub) + ZP
        target_emag = np.sqrt(ZP_err ** 2 + (2.5 / target_bkg_sub / np.log(10) * target_bkg_sub_err) ** 2)

        print(target_mag, target_emag)

        av_target_mag = np.average(target_mag, weights=1 / target_emag)
        av_target_mag_err = np.sqrt(np.cov(target_mag, aweights=1 / target_emag))

        print("K = %f +- %f mag" % (av_target_mag, av_target_mag_err))
        plt.show()

        # driz.write(f'data/{directory}/redux/combined/{target}.fits.gz')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection=new_wcs)
        # plt.imshow(driz.outsci, vmin=0, vmax=100)
        # plt.scatter(targ_coord.ra, targ_coord.dec, transform=ax.get_transform('world'), facecolors='none',
        #             edgecolors='r')
        # plt.show()


reduceNight('20220222', write_flat=False, write_flatten=False, write_bkg=False, write_bkg_sub=False)
