"""
13 January 2023
Seattle en route to Philly, Post-AAS241
Trevor McCaffrey

Running ICA on different data sets can be annoying because
just loading everything up in a notebook and the functions
take up loads of space, making it kind of a mess to work with.

Beyond that, it'd be good to share a script like this with
people like Keri and Gordon so they can quickly run ICA
without waiting on me.  Hopefully this works!
"""

#necessary imports - should only need these
#see Rankine+2020 for lmfit installation
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
import sys
#sys.path.append("/Users/Trevor1/Dropbox/ICA_module/")
sys.path.append("/Users/Trevor1/Dropbox/ICA_module/")
sys.path.append("/Users/Trevor1/Dropbox/HST/HSTCode/")
import plot_ICA
import spec_morph

def rebin(wave, flux, errs, mask):
    #create re-binned wavelength array - this is the easy part!
    #resolution is constant in log-space for SDSS, so easier to stick with that
    logwave = np.log10(wave)
    loglam_rebin = np.arange(logwave[0], logwave[-1], 0.0001)
    loglam_rebin_edges = loglam_rebin - 0.0001/2
    loglam_rebin_edges = np.append(loglam_rebin_edges, loglam_rebin[-1]+0.0001/2)

    #start empty
    flux_rebin = np.nan*np.zeros(len(loglam_rebin))
    errs_rebin = np.nan*np.zeros(len(loglam_rebin))
    mask_rebin = np.zeros(len(loglam_rebin))

    #and fill them in
    for i in range(len(loglam_rebin_edges)-1):
        ledge_rebin, redge_rebin = loglam_rebin_edges[i], loglam_rebin_edges[i+1]
        #orig edges
        #oldwidth =

        maskOrig = (logwave>ledge_rebin)&(logwave<redge_rebin)
        if not maskOrig.sum()==0:
            #flux_rebin[i] = ws.numpy_weighted_median(flux[maskOrig], weights=1./(errs[maskOrig]**2))
            flux_rebin[i] = np.median(flux[maskOrig])
            errs_rebin[i] = np.median(errs[maskOrig])
            mask_rebin[i] = np.max(mask[maskOrig])

    if (np.isnan(flux_rebin)).any():
        #initial resolution could be low enough such that some re-binned pixels weren't assigned a value
        #for those, interpolate with nearest neighbors
        for j in range(len(flux_rebin)):
            if np.isnan(flux_rebin[j]): #should make sure here it's not a known "gap", instead just ones missed from above
                arg1 = np.argmin(loglam_rebin[~np.isnan(flux_rebin)]-loglam[j])
                arg2 = arg2min(loglam_rebin[~np.isnan(flux_rebin)]-loglam[j])
                loglam1, flux1, errs1 = loglam_rebin[~np.isnan(flux_rebin)][arg1], \
                                        flux_rebin[~np.isnan(flux_rebin)][arg1], \
                                        errs_rebin[~np.isnan(flux_rebin)][arg1]
                loglam2, flux2, errs2 = loglam_rebin[~np.isnan(flux_rebin)][arg2], \
                                        flux_rebin[~np.isnan(flux_rebin)][arg2], \
                                        errs_rebin[~np.isnan(flux_rebin)][arg2]
                oldwidth = abs(loglam2-loglam1)
                #do linear fit on nearest non-nan neighbors
                m_flux, b_flux = np.polyfit([loglam1, loglam2], [flux1, flux2], 1)
                flux_rebin[j] = m_flux*loglam_rebin[j] + b_flux
                m_errs, b_errs = np.polyfit([loglam1, loglam2], [errs1, errs2], 1)
                errs_rebin[j] = (m_errs*loglam_rebin[j] + b_errs) * np.sqrt(oldwidth/0.0001)
                #leave mask[j]==0 for now

    wave_rebin = 10.**loglam_rebin
    return wave_rebin, flux_rebin, errs_rebin, mask_rebin

def blueshift(wave_half_flux):
    '''
    compute blueshift given wavelength where half the EW
    of CIV has been accumulated; units Angstroms
    '''
    return ((1549.48 - wave_half_flux) / 1549.48) * 3e5

def pixel_filter(wavelength,data,npix=20):

    filterval=npix//2
    filterval2=npix//2+1

    #print(len(data), data.dtype, data)
    indices = ~np.isnan(data) #indices we'll look at

    end        = data[indices][-filterval2-1:]
    end_wave   = wavelength[indices][-filterval2-1:]
    start      = data[indices][:filterval+1]
    start_wave = wavelength[indices][:filterval+1]
    right_median = np.nanmedian(end)
    left_median  = np.nanmedian(start)

    new_left_wave   = np.zeros(filterval)
    new_right_wave  = np.zeros(filterval)
    new_left_flux   = np.zeros(filterval)
    new_right_flux  = np.zeros(filterval)
    wave_devs_left  = np.zeros(filterval)
    wave_devs_right = np.zeros(filterval)
    flux_devs_left  = np.zeros(filterval)
    flux_devs_right = np.zeros(filterval)

    #Now flipping the mirrored spectra over the left and right medians:
    for k in range(filterval):
        wave_devs_left[-k-1]  = start_wave[k+1] - min(start_wave) #mirror across 0th pixel
        new_left_wave[-k-1]   = -2*wave_devs_left[-k-1] + start_wave[k+1]
        flux_devs_left[-k-1]  = start[k+1] - start[0] #left_median - TVM changed to edge pixel since I think that's right?
        new_left_flux[-k-1]   = -2*flux_devs_left[-k-1]+start[k+1] #want to add to mirrored array starting at end and moving left

        wave_devs_right[k]    = end_wave[-k-2] - max(end_wave)
        new_right_wave[k]     = -2*wave_devs_right[k] + end_wave[-k-2]
        flux_devs_right[k]    = end[-k-2] - end[-1] #right_median
        new_right_flux[k]     = -2*flux_devs_right[k]+end[-k-2]

    extended_data_array=np.concatenate((new_left_flux,data,new_right_flux),axis=0)
    continuum = np.array([np.nanmedian(extended_data_array[l-filterval:l+filterval]) for l in np.arange(filterval,len(data)+filterval,1)])

    return continuum

def maskNAL(wave, flux, errs, mask):
    mask_wnal = mask.copy()
    flux_median_61p = pixel_filter(wave, flux, 61)
    mask_wnal[(flux_median_61p-flux > 3*errs)] = 2.
    #Pad the NAL mask three pixels each side
    for i in range(len(wave)):
        if mask_wnal[i]==2.:
            mask_wnal[i-3:i+3+1] = 3. #this keeps from masking the entire spectrum in the for loop!
    mask_wnal[mask_wnal==3] = 2 #save them all as two
    return mask_wnal

def get_f2500(wave, flux, errs, mask, norm_coeff, ica_path="./"):
    #real units!  also using the final generated mask
    wave_norm_ica, flux_norm_ica = get_ICA(wave, flux, errs, mask,
                                 ica_path=ica_path,
                                 plot_spectrum=False, CHISQ=True, comps_use=None)
    ind2500 = np.argmin(abs(wave_norm_ica-2500.))
    return flux_norm_ica[ind2500]*norm_coeff

def load_ICA(ica_path="./"):
    """
    load in the ICA components; there are three sets:
    (1) amy_12603000_10c_180421.comp are the 10 "main" components
    (2) amy_12753000_lowew_10c_181101_v1.comp are 10 different components
        which typically work better on low EW spectra when the main components fail
    (3) amy_12653000_hew_hsn_7c_190302.comp are 7 components which typically work
        better on high EW spectra when the main components fail

    Note the difference in wavelength coverage for each set of components!

    First load components in as a pandas dataframe, then turn them into
    (Ncomponents)x(Npixels) arrays.
    """
    wave_modEW          = pd.read_csv(ica_path+"wav_12603000.dat", names=["wave"]).values.flatten()
    components_modEW_df = pd.read_csv(ica_path+"amy_12603000_10c_180421.comp", sep="\s+",
                          names=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"])
    components_modEW    = np.array([components_modEW_df.iloc[:,i].values for i in range(components_modEW_df.shape[1])])

    wave_lowEW          = pd.read_csv(ica_path+"wav_12753000.dat", names=["wave"]).values.flatten()
    components_lowEW_df = pd.read_csv(ica_path+"amy_12753000_lowew_10c_181101_v1.comp", sep="\s+",
                          names=["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"])
    components_lowEW    = np.array([components_lowEW_df.iloc[:,i].values for i in range(components_lowEW_df.shape[1])])

    wave_highEW          = pd.read_csv(ica_path+"wav_12653000.dat", names=["wave"]).values.flatten()
    components_highEW_df = pd.read_csv(ica_path+"amy_12653000_hew_hsn_7c_190302.comp", sep="\s+",
                          names=["c1","c2","c3","c4","c5","c6","c7"])
    components_highEW    = np.array([components_highEW_df.iloc[:,i].values for i in range(components_highEW_df.shape[1])])

    return wave_modEW, components_modEW, wave_lowEW, components_lowEW, wave_highEW, components_highEW

def residual(params, comps, data, eps_data):
    '''
    compute residuals between data and model,
    weighted by the variance array;
    used by minimize() function in ICA_fit()
    '''
    model = sum(params[w].value*comps[n] for (n, w) in enumerate(params))
    return (data-model)**2 * eps_data

def ICA_fit(components, wave, spectrum, ivar, flags):
    # FIXME: explain this
    #this is called ICA_fit, but we get the component weights from minimize()
    '''
    Instansiate Parameters and add empty weight values;
    then compute the ICA weights using minimize()
    '''
    params = Parameters()
    for i in range(len(components)):
        params.add('W%d' % (i+1), value=0)
    #badpix = masking(wave, spectrum, 1/np.sqrt(ivar))
    ivar[flags>0] = 0.
    return minimize(residual, params, args=(components, spectrum, ivar), method="lbfgsb", nan_policy='omit')

def get_ICA(wave, flux, errs, mask, ica_path="./", plot_spectrum=True, CHISQ=True, comps_use=None):
    #most of this function just aligns the pixels from your spectrum and the ICA components
    wave_mod, components_mod, wave_low, components_low, wave_high, components_high = load_ICA(ica_path)

    #Apply mask and get fits
    mod_mask  = (wave>1260)&(wave<3000)
    low_mask  = (wave>1275)&(wave<3000)
    high_mask = (wave>1265)&(wave<3000)

    mod_comp_mask  = (wave_mod>min(wave[mod_mask]))&(wave_mod<max(wave[mod_mask]))
    low_comp_mask  = (wave_low>min(wave[low_mask]))&(wave_low<max(wave[low_mask]))
    high_comp_mask = (wave_high>min(wave[high_mask]))&(wave_high<max(wave[high_mask]))

    #print(mod_comp_mask.shape)
    ###########
    #Adjust array sizes
    wave_mod_fit        = wave_mod[mod_comp_mask]
    components_mod_fit  = components_mod[:,mod_comp_mask]
    wave_low_fit        = wave_low[low_comp_mask]
    components_low_fit  = components_low[:,low_comp_mask]
    wave_high_fit       = wave_high[high_comp_mask]
    components_high_fit = components_high[:,high_comp_mask]

    ###############
    #Now do fitting

    # mod EW
    #print(len(wave[mod_mask]), len(wave_mod_fit))
    if len(wave[mod_mask])==len(wave_mod_fit):
        fit_mod  = ICA_fit(components_mod_fit, wave[mod_mask], flux[mod_mask], 1/(errs[mod_mask]**2), mask[mod_mask])
    else:
        #input wave is greater by one pixel, cut components byb one
        if len(wave[mod_mask])-len(wave_mod_fit) == -1:
            fit_mod  = ICA_fit(components_mod_fit[:,:-1], wave[mod_mask], flux[mod_mask], 1/(errs[mod_mask]**2), mask[mod_mask])
        elif len(wave[mod_mask])-len(wave_mod_fit) == 1:
            fit_mod  = ICA_fit(components_mod_fit, wave[mod_mask][:-1], flux[mod_mask][:-1], 1/(errs[mod_mask][:-1]**2), mask[mod_mask][:-1])
    #low EW
    if len(wave[low_mask])==len(wave_low_fit):
        fit_low  = ICA_fit(components_low_fit, wave[low_mask], flux[low_mask], 1/(errs[low_mask]**2), mask[low_mask])
    else:
        #input wave is greater by one pixel, cut components byb one
        if len(wave[low_mask])-len(wave_low_fit) == -1:
            fit_low  = ICA_fit(components_low_fit[:,:-1], wave[low_mask], flux[low_mask], 1/(errs[low_mask]**2), mask[low_mask])
        elif len(wave[low_mask])-len(wave_low_fit) == 1:
            fit_low  = ICA_fit(components_low_fit, wave[low_mask][:-1], flux[low_mask][:-1], 1/(errs[low_mask][:-1]**2), mask[low_mask][:-1])

    #high EW
    if len(wave[high_mask])==len(wave_high_fit):
        fit_high  = ICA_fit(components_high_fit, wave[high_mask], flux[high_mask], 1/(errs[high_mask]**2), mask[high_mask])
    else:
        #input wave is greater by one pixel, cut components byb one
        if len(wave[high_mask])-len(wave_high_fit) == -1:
            fit_high  = ICA_fit(components_high_fit[:,:-1], wave[high_mask], flux[high_mask], 1/(errs[high_mask]**2), mask[high_mask])
        elif len(wave[high_mask])-len(wave_high_fit) == 1:
            #print(wave_high_fit.shape, wave[high_mask].shape)
            fit_high  = ICA_fit(components_high_fit, wave[high_mask][:-1], flux[high_mask][:-1], 1/(errs[high_mask][:-1]**2), mask[high_mask][:-1])

    #this is what all branches above would look like w/o index problems
    #fit_mod   = ICA_fit(components_mod[:,mod_comp_mask], wave[mod_mask], flux[mod_mask], 1/(errs[mod_mask]**2), mask[mod_mask]) if len(wave[mod_mask])==len(wave_mod) else ICA_fit(components_mod[:,:-1], wave[mod_mask], flux[mod_mask], 1/(errs[mod_mask]**2), mask[mod_mask])
    #fit_low   = ICA_fit(components_low[:,mod_comp_mask], wave[low_mask], flux[low_mask], 1/(errs[low_mask]**2), mask[low_mask]) if len(wave[low_mask])==len(wave_low) else ICA_fit(components_low[:,:-1], wave[low_mask], flux[low_mask], 1/(errs[low_mask]**2), mask[low_mask])
    #fit_high  = ICA_fit(components_high[:,mod_comp_mask], wave[high_mask], flux[high_mask], 1/(errs[high_mask]**2), mask[high_mask]) if len(wave[high_mask])==len(wave_high) else ICA_fit(components_high[:,:-1], wave[high_mask], flux[high_mask], 1/(errs[high_mask]**2), mask[high_mask])

    #compute weights
    weights_mod  = [fit_mod.params[i].value for i in fit_mod.params]
    weights_low  = [fit_low.params[i].value for i in fit_low.params]
    weights_high = [fit_high.params[i].value for i in fit_high.params]

    if CHISQ:
        if fit_mod.redchi < fit_high.redchi and fit_mod.redchi < fit_low.redchi:
            return wave_mod, np.dot(weights_mod, components_mod)
        elif fit_low.redchi < fit_high.redchi and fit_low.redchi < fit_mod.redchi:
            return wave_low, np.dot(weights_low, components_low)
        else:
            return wave_high, np.dot(weights_high, components_high)

    #enter this branch if you want to tell ICA which components to use
    else:
        if comps_use=="mod":
            return wave_mod, np.dot(weights_mod, components_mod)
        elif comps_use=="low":
            return wave_low, np.dot(weights_low, components_low)
        elif comps_use=="high":
            return wave_high, np.dot(weights_high, components_high)
        else:
            print("Didn't recognize comps_use")

def maskIterate(wave, flux, errs, mask, wave_ica, flux_ica, ica_path="/Users/trevormccaffrey/Dropbox/ICA_module/components/"):
    #Iterative masking for narrow absorption
    #It seems that, without this step, the reconstructions are in general underestimated
    #Start: if (reconstruction-flux)>N*sigma, mask this pixel
    N = 2.0
    #Break here if N > 4
    while N <= 4:
        #Start from scratch with original wave, flux, etc
        niter = 0
        while niter < 10:
            nbad = 0
            #Work out the window from full spec we're working on
            wave_begin = max( min(wave_ica), min(wave) )
            wave_end   = min( max(wave_ica), max(wave) )
            ind_begin = np.argmin( abs(wave-wave_begin) )
            ind_end   = np.argmin( abs(wave-wave_end) )

            #Do the actual masking
            for i in range(ind_begin, ind_end+1):
                #ind_reconst = np.argmin( abs(wave[i]-wave_ica) )
                i_left  = max(ind_begin, i-15)
                i_right = min(ind_end, i+15)
                ir_left  = np.argmin( abs(wave[i_left]-wave_ica) )
                ir_right = np.argmin( abs(wave[i_right]-wave_ica) )
                #change here to only mask that pixel for now;
                #should only mask the 10 nearest after the initial bad pixels are established
                badpix = (flux_ica[ir_left:ir_right+1]-flux[i_left:i_right+1]) > N*errs[i_left:i_right+1]
                if badpix.sum() > (len(badpix)//2) and mask[i]==0:# and ~em[i]:
                    nbad += 1
                    #mask[max(ind_begin,i-10):min(ind_end,i+11)] = 1
                    mask[i] = 4

            #If no badpix, mask has converged and we can break
            if nbad == 0:
                break
            else:
                niter += 1
                #print("N = %.2f, Try Number %d" % (N,niter))
                wave_ica, flux_ica = get_ICA(wave, flux, errs, mask, name="",
                                              ica_path=ica_path, plot_spectrum=False, CHISQ=True, comps_use=None)
                N += 0.25
        if nbad == 0:
            #now that individually masked pixels are established,
            #mask 10 additional pixels redward and blueward of those
            for i in range(len(mask)):
                if mask[i]==4:
                    mask[i-10:i+11] = 4
            break

    return mask

def main_ICA(waveSpec, fluxSpec, errsSpec, maskSpec, z, name="", ica_path="./"):
    wave = waveSpec.copy()
    flux = fluxSpec.copy()
    errs = errsSpec.copy()
    mask = maskSpec.copy()

    # check wavelength resolution, and rebin if it doesn't match components (~69 km/s)
    resolution = np.nanmedian([3.e5*((wave[i+1]-wave[i])/wave[i]) for i in range(len(wave)-1)])
    if round(resolution)!=69:
        print("Input wavelength resolution does not match 69 km/s.  Re-binning spectrum.")
        wave, flux, errs, mask = rebin(wave, flux, errs, mask)

    #normalize spectrum
    norm_coeff = np.nanmedian(flux)
    flux /= norm_coeff
    errs /= norm_coeff

    #Mask pixels >3sigma below pseudo continuum
    #initial mask is just from initial pipeline (e.g. SDSS, HST)
    mask = maskNAL(wave, flux, errs, mask)

    #standard-ize spectrum shape by morphing
    flux_morph, morph_coeff = spec_morph.morph2(wave*(1+z), flux, errs, z, "")
    errs_morph = errs * morph_coeff

    #do initial fit without BAL mask
    wave_init_ica, flux_init_ica = get_ICA(wave, flux_morph, errs_morph, mask, ica_path)

    #generate iterative BAL mask
    ## FIXME: for BAL quasars, can start with either manually defined mask covering troughs
    ##        or generic definition done in Rankine+20
    ##        don't want to mask generic BAL troughs for non-BALs though
    mask_witer = maskIterate(wave, flux_morph, errs_morph, mask,
                             wave_init_ica, flux_init_ica)

    #now do ICA fit with final mask applied
    wave_ica, flux_ica = get_ICA(wave, flux_morph, errs_morph, mask_witer,
                                 ica_path=ica_path,
                                 plot_spectrum=False, CHISQ=True, comps_use=None)

    # FIXME: should add code here that checks if CIII] is in spectrum,
    # if it is, measure blueshift and generate priors for weights based on measurement

    #get f2500 from reconstruction
    #de-morph and de-normalize the spectrum to get back to real units
    # FIXME: need to get indexing right here for edge cases - commenting out for now
    try:
        iwave_start = np.argmin(abs(wave-min(wave_ica)))
        flux_ica_realunits = flux_ica / morph_coeff[iwave_start:iwave_start+len(wave_ica)] * norm_coeff
        f2500_ica = flux_ica_realunits[np.argmin(abs(wave_ica-2500.))]
    #Note the input units - normalized, but not morphed
    except ValueError:
        f2500_ica = get_f2500(wave, flux, errs, mask_witer, norm_coeff, ica_path=ica_path)

    return wave, flux_morph, errs_morph, mask_witer, wave_ica, flux_ica, f2500_ica #return flux with Arbitrary units as well for plotting
