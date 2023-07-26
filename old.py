import os

import astropy.stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
mpl.use('macosx')
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from drizzle.drizzle import Drizzle
from photutils.detection import DAOStarFinder
import sep
from photutils.aperture import CircularAperture
import astroalign as aa
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

def stack_images(directory):
    if not os.path.exists(f'backgrounds/{directory}'):
        os.mkdir(f'backgrounds/{directory}')
    file_list = np.sort(os.listdir(directory))
    file_list = [file for file in file_list if '.fits.gz' in file]
    # Load all data for given night--------------------------------------------
    all_targets = []
    all_imgs = []
    all_flats = []
    all_wcs = []
    all_heads = []
    for i, file in enumerate(file_list):
        with fits.open(os.path.join(directory, file)) as hdu:
            head = hdu[0].header
            img = hdu[0].data
        target = head['TARGNAME']
        all_imgs.append(img)  # [::-1, ::-1])  # np.rot90(img)
        all_heads.append(head)
        all_targets.append(target)
        all_wcs.append(WCS(head))
    all_targets = np.array(all_targets)
    all_imgs = np.array(all_imgs)
    # Calculate and subtract flats by median combining all images
    flat = np.median(all_imgs, axis=0)
    flat /= np.median(flat)

    # Stop bad pixels blowing up
    flat[flat < 0.5] = 1
    hdu = fits.PrimaryHDU(flat)
    hdul = fits.HDUList([hdu])
    hdul.writeto(f'flats/{directory}.fits', overwrite=True)
    all_imgs /= flat[None, ...]

    # Subtract sky from data from each image
    for target in np.unique(all_targets):
        if target[0:3] != '202': # Skip non-SNe
            continue
        if target == '2023ixf':
            continue
        if target in ['2022aaiq_S2']:
            continue
        if target != '2023emq':
            continue
        print(target)
        target_inds = all_targets == target
        #extra_target_inds = all_targets == f'{target}_S1'
        #target_inds = target_inds + extra_target_inds
        start, stop = np.where(target_inds)[0][0], np.where(target_inds)[0][-1]
        target_imgs = all_imgs[target_inds, ...]
        target_heads = all_heads[start: stop + 1]
        if target_imgs.shape[0] < 2:
            continue
        offsets = np.zeros(len(target_heads))
        for i, head in enumerate(target_heads):
            offsets[i] = np.abs(head['XOFFSET']) + np.abs(head['YOFFSET'])
        start, stop = np.where(offsets == 0)[0][0], np.where(offsets == 0)[0][-1]
        # start, stop = 17, 23

        """for img in phot_imgs:
            plt.figure()
            vmin, vmax = np.quantile(img.flatten(), [0.1, 0.9])
            plt.imshow(img, vmin=vmin, vmax=vmax)
        plt.show()"""

        # Build mask to avoid including sources in background
        background_scales = []
        mask = np.zeros_like(target_imgs)
        for i, img in enumerate(target_imgs):
            mean, median, std = sigma_clipped_stats(img)
            mask[i, ...] = img > median + 10 * std
            background_scale = np.nanmedian(img.flatten())
            background_scales.append(background_scale)
        background_scales = np.array(background_scales)
        ratios = background_scales / np.median(background_scales)

        bg = np.median(target_imgs, axis=0)
        bg_array = np.ma.array(target_imgs, mask=mask)
        mask_bg = np.ma.median(bg_array, axis=0)
        bg = mask_bg * (1 - mask_bg.mask) + mask_bg.mask * bg

        #phot_imgs = target_imgs[start: stop + 1]
        #bg = np.median(phot_imgs, axis=0)
        #bg = np.nanmedian(astropy.stats.sigma_clip(target_imgs, mas, axis=None, sigma_upper=3, sigma_lower=10), axis=0)
        hdul = fits.HDUList([hdu])
        hdul.writeto(f'backgrounds/{directory}/{target}.fits', overwrite=True)
        bg = bg[None, ...] * ratios[:, None, None]
        target_imgs -= bg

        """for img in phot_imgs:
            plt.figure()
            vmin, vmax = np.quantile(img.flatten(), [0.1, 0.9])
            plt.imshow(img, vmin=vmin, vmax=vmax)
        plt.show()
        return"""

        phot_imgs = target_imgs[start: stop + 1]  # Only use images where source is not on slit for photometry
        phot_heads = target_heads[start: stop + 1]

        # Fix WCS with rotation
        for i in range(target_imgs.shape[0]):
            rot = 180 - target_heads[i]['ROTDEST']
            CD = np.array([[target_heads[i]['CD1_1'], target_heads[i]['CD1_2']], [target_heads[i]['CD2_1'], target_heads[i]['CD2_2']]])
            R = np.array([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))], [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))]])
            CD_new = R @ CD
            target_heads[i]['CD1_1'] = CD_new[0, 0]
            target_heads[i]['CD1_2'] = CD_new[0, 1]
            target_heads[i]['CD2_1'] = CD_new[1, 0]
            target_heads[i]['CD2_2'] = CD_new[0, 1]

        # Correct images for WCS
        for i in range(target_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[0]))
            driz.add_image(target_imgs[i], WCS(target_heads[i]))
            target_imgs[i] = driz.outsci
            plt.figure()
            plt.imshow(driz.outsci, vmin=-50, vmax=200)
        plt.show()
        return
        # Find brightest source in first image
        mean, median, std = sigma_clipped_stats(target_imgs[0])
        daofind = DAOStarFinder(fwhm=6, threshold=10 * std)
        sources = daofind(target_imgs[0]).to_pandas().sort_values(by='flux', ascending=False)
        try:
            max_source = sources.iloc[0]
        except:
            continue
        ref_x, ref_y = max_source.xcentroid, max_source.ycentroid
        # Find x/y shifts
        xs, ys = [], []
        for i in range(phot_imgs.shape[0]):
            mean, median, std = sigma_clipped_stats(phot_imgs[i])
            daofind = DAOStarFinder(fwhm=6, threshold=5*std)
            #try:
            sources = daofind(phot_imgs[i]).to_pandas().sort_values(by='flux', ascending=False)
            sources['sep'] = np.sqrt((sources.xcentroid - ref_x)**2 + (sources.ycentroid - ref_y)**2)
            search_rad = 2000  # Should really be smaller but lack of WCS alignment means big radius necessary
            sources = sources[np.sqrt((sources.xcentroid - ref_x)**2 + (sources.ycentroid - ref_y)**2) < search_rad]
            max_source = sources.iloc[0]
            xs.append(max_source.xcentroid)
            ys.append(max_source.ycentroid)
        #     print(sources)
        #     plt.figure()
        #     positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        #     apertures = CircularAperture(positions, r=8.0)
        #     plt.imshow(phot_imgs[i], cmap='gray', vmin=-50, vmax=200)
        #     apertures.plot(color='blue', lw=2.5, alpha=0.5)
        #     plt.show()
        # return
        #wcs = WCS(target_heads[0])
        #driz = Drizzle(outwcs=WCS(target_heads[0]))
        #for i in range(phot_imgs.shape[0]):
        #    driz.add_image(target_imgs[i], WCS(target_heads[i]))
        #plt.imshow(driz.outsci, vmin=-50, vmax=200)
        #plt.figure()
        for i in range(phot_imgs.shape[0]):
            xshift, yshift = xs[i] - xs[0], ys[i] - ys[0]
            x_scale, y_scale = -1, -1
            cosdec = np.cos(np.deg2rad(phot_heads[0]['CRVAL2']))
            phot_heads[i]['CRVAL1'] = phot_heads[0]['CRVAL1'] + (xshift * phot_heads[0]['CD1_1'] * x_scale + yshift * phot_heads[0]['CD1_2'] * y_scale) / cosdec
            phot_heads[i]['CRVAL2'] = phot_heads[0]['CRVAL2'] + xshift * phot_heads[0]['CD2_1'] * x_scale + yshift * phot_heads[0]['CD2_2'] * y_scale
        # Save intermediate images
        for i in range(phot_imgs.shape[0]):
            driz = Drizzle(outwcs=WCS(target_heads[i]))
            driz.add_image(phot_imgs[i], WCS(target_heads[i]))
            driz.write(f'processed_images/{target}_{i}.fits')
        # Save output
        driz = Drizzle(outwcs=WCS(target_heads[0]))
        for i in range(phot_imgs.shape[0]):
            driz.add_image(phot_imgs[i], WCS(phot_heads[i]))
        #plt.imshow(driz.outsci, vmin=-50, vmax=200)
        #plt.show()
        # Do photometry
        bkg = sep.Background(driz.outsci)
        objects = sep.extract(driz.outsci, 3, err=bkg.globalrms)

        # plot background-subtracted image
        fig, ax = plt.subplots()
        m, s = np.mean(driz.outsci), np.std(driz.outsci)
        im = ax.imshow(driz.outsci, interpolation='nearest', cmap='gray',
                       vmin=m - s, vmax=m + s, origin='lower')

        # plot an ellipse for each object
        for i in range(len(objects)):
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=6 * objects['a'][i],
                        height=6 * objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)
        plt.show()

        if not os.path.exists(f'data/output/{directory}'):
            os.mkdir(f'data/output/{directory}')
        driz.write(f'output/{directory}/{target}.fits')
        return

stack_images('20230410')

