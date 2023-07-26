import os, glob, copy

import astropy.stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy

mpl.use('macosx')
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from drizzle.drizzle import Drizzle
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
import astroalign as aa
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

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


def reduceNight(directory, write_flat=True, write_bkg=True, \
                write_flatten=True, write_bkg_sub=True, \
                verbose=True):
    """
    This is the main function used to reduce an entire night of data.
    Input:
        directory:      string of the path to the raw NIRES SVC images
        write_flat:     bool write master flat for the night
        write_bkg:      bool write background frame for each target
        write_flatten:  bool write flat-corrected science images
        write_bkg_sub:  bool write the individual background subtracted sci images
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
        #     if target[0:3] != '202': # Skip non-SNe
        #         continue
        if target not in targets_to_process:
            continue  # as in quit the loop and continue
        if target in ['13232490+444059']:
             continue
        if target != '2023gfo':
            continue
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
        # Figure out which images have the source outside of the slit
        for i, head in enumerate(target_heads):
            offsets[i] = np.abs(head['XOFFSET']) + np.abs(head['YOFFSET'])
        #     print(offsets)
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
        for i in range(target_imgs.shape[0]):#range(phot_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[0]))
            driz.add_image(target_imgs[i], WCS(target_heads[i]))
            target_imgs[i] = driz.outsci

        # ##########FINE ALIGNMENT FROM OLD SCRIPT###################

        # # Image shift using cross-correlations-------------------------------
        # # Remove negative sources, just for cross-correlation
        # align_images = target_imgs.copy()
        # align_images[align_images < 0] = 0
        #
        # # This will work for an image containing only point sources, but will struggle with extended sources, need to
        # # improve
        # self_corr = scipy.signal.correlate(align_images[0], align_images[0], mode='same')
        # ref1 = np.unravel_index(np.argmax(self_corr), self_corr.shape)
        # for i in range(phot_imgs.shape[0]):
        #     cross_corr = scipy.signal.correlate(align_images[0][:412, 612:], align_images[i][:412, 612:], mode='same')
        #     ref2 = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        #     xshift, yshift = np.array(ref2) - np.array(ref1)
        #     x_scale, y_scale = 1, 1
        #     cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
        #     target_heads[i]['CRVAL1'] = target_heads[0]['CRVAL1'] + (
        #                 xshift * target_heads[0]['CD1_1'] * x_scale + yshift * target_heads[0][
        #             'CD1_2'] * y_scale) / cosdec
        #     target_heads[i]['CRVAL2'] = target_heads[0]['CRVAL2'] + xshift * target_heads[0][
        #         'CD2_1'] * x_scale + yshift * target_heads[0]['CD2_2'] * y_scale

        # Image shift using reference source-----------------------------

        # # Find reference sources in first image
        mean, median, std = sigma_clipped_stats(target_imgs[0][~bpm])

        #use bpm here
        daofind = DAOStarFinder(fwhm=8, threshold=10 * std)
        sources = daofind(target_imgs[0], mask=bpm).to_pandas().sort_values(by='flux', ascending=False)
        # Ignore sources too close to edge
        sources = sources[(sources['xcentroid'] > 50) & (sources['xcentroid'] < 974) &
                          (sources['ycentroid'] > 50) & (sources['ycentroid'] < 974)]
        # Ignore very faint sources that are likely just noise
        # sources = sources[sources['mag'] < -2]
        max_source = sources.iloc[0]
        ref_x, ref_y = max_source.xcentroid, max_source.ycentroid
        # plt.figure()
        # positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        # apertures = CircularAperture(positions, r=8.0)
        # plt.imshow(target_imgs[0], cmap='gray', vmin=-50, vmax=500)
        # apertures.plot(color='blue', lw=2.5, alpha=0.5)
        # plt.show()
        # Find x/y shifts
        xs, ys = [], []
        for i in range(phot_imgs.shape[0]):
            mean, median, std = sigma_clipped_stats(target_imgs[i][~bpm])
            daofind = DAOStarFinder(fwhm=10, threshold=1*std)
            sources = daofind(target_imgs[i], mask=bpm).to_pandas().sort_values(by='flux', ascending=False)
            sources['sep'] = np.sqrt((sources.xcentroid - ref_x)**2 + (sources.ycentroid - ref_y)**2)
            search_rad = 30
            sources = sources[np.sqrt((sources.xcentroid - ref_x)**2 + (sources.ycentroid - ref_y)**2) < search_rad]
            max_source = sources.iloc[0]
            xs.append(max_source.xcentroid)
            ys.append(max_source.ycentroid)
            # print(sources)
            # plt.figure()
            # positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            # apertures = CircularAperture(positions, r=8.0)
            # plt.imshow(target_imgs[i], cmap='gray', vmin=-50, vmax=500)
            # apertures.plot(color='blue', lw=2.5, alpha=0.5)
            # plt.show()
        for i in range(phot_imgs.shape[0]):
            xshift, yshift = xs[i] - xs[0], ys[i] - ys[0]
            print(i, xshift, yshift)
            x_scale, y_scale = -1, -1
            cosdec = np.cos(np.deg2rad(target_heads[0]['CRVAL2']))
            target_heads[i]['CRVAL1'] = target_heads[0]['CRVAL1'] + (xshift * target_heads[0]['CD1_1'] * x_scale + yshift * target_heads[0]['CD1_2'] * y_scale) / cosdec
            target_heads[i]['CRVAL2'] = target_heads[0]['CRVAL2'] + xshift * target_heads[0]['CD2_1'] * x_scale + yshift * target_heads[0]['CD2_2'] * y_scale
            # Save intermediate images
        # except:
        #     print("Fine alignment fails. Probably no sources found! Skipping for now")
        #     continue

        for i in range(phot_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[i]))
            driz.add_image(target_imgs[i], WCS(target_heads[i]))
            driz.write(f'data/{directory}/redux/{target}_{i}.fits.gz')
        # Save output
        driz = Drizzle(outwcs=WCS(target_heads[0]))
        for i in range(phot_imgs.shape[0]):
            driz.add_image(target_imgs[i], WCS(target_heads[i]))

        driz.write(f'data/{directory}/redux/combined/{target}.fits.gz')
        return


reduceNight('20230607', write_flat=False, write_flatten=False, write_bkg=False, write_bkg_sub=False)







