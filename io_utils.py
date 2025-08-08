def append_rates(path, outfilename, detection_rate, formation_rate, merger_rate, redshifts, COMPAS, Average_SF_mass_needed, shell_volumes, n_redshifts_detection,
    maxz=5., sensitivity="O1", dco_type="BHBH", mu0=0.035, muz=-0.23, sigma0=0.39, sigmaz=0., alpha=0., aSF = 0.01, bSF = 2.77, cSF = 2.90, dSF = 4.70,
    append_binned_by_z = False, redshift_binsize=0.1):
    """
        Append the formation rate, merger rate, detection rate and redshifts as a new group to your COMPAS output with weights hdf5 file

        Args:
            path                   --> [string] Path to the COMPAS file that contains the output
            outfilename            --> [string] Name of the hdf5 file that you want to write your rates to
            detection_rate         --> [2D float array] Detection rate for each binary at each redshift in 1/yr
            formation_rate         --> [2D float array] Formation rate for each binary at each redshift in 1/yr/Gpc^3
            merger_rate            --> [2D float array] Merger rate for each binary at each redshift in 1/yr/Gpc^3
            redshifts              --> [list of floats] List of redshifts
            COMPAS                 --> [Object]         Relevant COMPAS data in COMPASData Class
            Average_SF_mass_needed --> [float]          How much star forming mass your simulation represents on average
            shell_volumes          --> [list of floats] Equivalent of redshifts but converted to shell volumes
            n_redshifts_detection  --> [int]            Number of redshifts in list that should be used to calculate detection rates

            maxz                   --> [float] Maximum redshhift up to where we would like to store the data
            sensitivity            --> [string] Which detector sensitivity you used to calculate rates 
            dco_type               --> [string] Which DCO type you used to calculate rates 
            mu0                    --> [float]  metallicity dist: expected value at redshift 0
            muz                    --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                 --> [float]  metallicity dist: width at redshhift 0
            sigmaz                 --> [float]  metallicity dist: redshift evolution of width
            alpha                  --> [float]  metallicity dist: skewness (0 = lognormal)

            append_binned_by_z     --> [Bool] to save space, bin rates by redshiftbin and append binned rates
            redshift_binsize       --> [float] if append_binned_by_z, how big should your redshift bin be

        Returns:
            h_new                  --> [hdf5 file] Compas output file with a new group "rates" with the same shape as DoubleCompactObjects x redshifts
    """
    print('\nIn append rates: shape redshifts', np.shape(redshifts))
    print('shape COMPAS.sw_weights', np.shape(COMPAS.sw_weights) )
    print('COMPAS.DCOmask', COMPAS.DCOmask, ' was set for dco_type', dco_type )
    print('shape COMPAS COMPAS.DCOmask', np.shape(COMPAS.DCOmask), ' sums to ', np.sum(COMPAS.DCOmask) )
    print('path', path)

    #################################################
    #Open hdf5 file that we will read from
    print('path', path)
    with h5.File(path , 'r') as f_COMPAS:
        
        # Would you like to write your rates to a different file? 
        if path == outfilename:
            raise ValueError('you cant append directly to the input data, will change outfilename to %s'%(outfilename)+'_1')
            outfilename = outfilename+'_1'

        #'you want to save your output to a different file!'
        if os.path.exists(outfilename):
            print('file', outfilename, 'exists!! You will remove it')
            os.remove(outfilename)
            
        print('writing to ', outfilename)
        h_new = h5.File(outfilename, 'w')

        # The rate info is shaped as BSE_Double_Compact_Objects[COMPAS.DCOmask] , len(redshifts)
        try: 
            DCO             = f_COMPAS['BSE_Double_Compact_Objects']#
        except:
            DCO             = f_COMPAS['DoubleCompactObjects']#

        #################################################
        # Create a new group where we will store data
        new_rate_group = 'Rates_mu0{}_muz{}_alpha{}_sigma0{}_sigmaz{}_a{}_b{}_c{}_d{}'.format(mu0, muz, alpha, sigma0, sigmaz, aSF, bSF, cSF, dSF)

        if append_binned_by_z:
            new_rate_group  = new_rate_group + '_zBinned'

        if new_rate_group not in h_new:
            h_new.create_group(new_rate_group)
        else:
            print(new_rate_group, 'exists, we will overrwrite the data')


        #################################################
        # Bin rates by redshifts
        #################################################
        if append_binned_by_z:
            # Choose how you want to bin the redshift, these represent the left and right boundaries
            redshift_bins = np.arange(0, redshifts[-1]+redshift_binsize, redshift_binsize)
            print('new crude redshift_bins', redshift_bins)
            print('old fine redshifts', redshifts)
            fine_binsize    = np.diff(redshifts)[0] #Assunming your redshift bins are equally spaced!!
            print('fine_binsize', fine_binsize)
            #Assuming your crude redshift bin is made up of an integer number of fine z-bins!!!
            i_per_crude_bin = redshift_binsize/fine_binsize 
            print('i_per_crude_bin', i_per_crude_bin)
            i_per_crude_bin = int(i_per_crude_bin)

            ###################
            # convert crude redshift bins to volumnes and ensure all volumes are in Gpc^3
            crude_volumes = cosmology.comoving_volume(redshift_bins).to(u.Gpc**3).value
            # split volumes into shells and duplicate last shell to keep same length
            crude_shell_volumes    = np.diff(crude_volumes)
            # crude_shell_volumes    = np.append(crude_shell_volumes, crude_shell_volumes[-1])

            ###################
            # convert redshifts to volumnes and ensure all volumes are in Gpc^3
            fine_volumes       = cosmology.comoving_volume(redshifts).to(u.Gpc**3).value
            fine_shell_volumes = np.diff(fine_volumes)
            fine_shell_volumes = np.append(fine_shell_volumes, fine_shell_volumes[-1])

            # Use digitize to assign the redshifts to a bin (detection list is shorter)
            # digitized     = np.digitize(redshifts, redshift_bins)
            digitized_det = np.digitize(redshifts[:n_redshifts_detection], redshift_bins)

            # Convert your merger_rate back to 1/yr by multiplying by the fine_shell_volumes
            N_dco_in_z_bin      = (merger_rate[:,:] * fine_shell_volumes[:])
            print('fine_shell_volumes', fine_shell_volumes)

            # The number of merging BBHs that need a weight
            N_dco  = len(merger_rate[:,0])
            
            ####################
            # binned_merger_rate will be the (observed) weights, binned by redshhift
            binned_merger_rate    = np.zeros( (N_dco, len(redshift_bins)-1) )# create an empty list to fill
            binned_detection_rate = np.zeros( (N_dco, len(redshift_bins)-1) )# create an empty list to fill

            # loop over all redshift redshift_bins
            for i in range(len(redshift_bins)-1):
                # print('redshifts[Bool_list[i]]', redshifts[Bool_list[i]])
                print('redshifts[[i*i_per_crude_bin:(i+1)*i_per_crude_bin]]', redshifts[i*i_per_crude_bin:(i+1)*i_per_crude_bin])

                # Sum the number of mergers per year, and divide by the new dz volume to get a density
                binned_merger_rate[:,i] = np.sum(N_dco_in_z_bin[:,i*i_per_crude_bin:(i+1)*i_per_crude_bin], axis = 1)/crude_shell_volumes[i]

                # only add detected rates for the 'detectable' redshifts
                if redshift_bins[i] < redshifts[n_redshifts_detection]:
                    # The detection rate was already multiplied by the shell volumes, so we can sum it directly
                    binned_detection_rate[:,i] = np.sum(detection_rate[:, digitized_det == i+1], axis = 1)

            #  To avoid huge filesizes, we don't really wan't All the data, 
            # so we're going to save up to some redshift
            z_index = np.digitize(maxz, redshift_bins) -1

            # The detection_rate is a smaller array, make sure you don't go beyond the end
            detection_index = z_index if z_index < n_redshifts_detection else n_redshifts_detection
            
            save_redshifts        = redshift_bins[:z_index]
            save_merger_rate      = binned_merger_rate[:,:z_index]
            # save_detection_rate   = binned_detection_rate[:,:detection_index]

        else: 
            #  To avoid huge filesizes, we don't really wan't All the data, 
            # so we're going to save up to some redshift
            z_index = np.digitize(maxz, redshifts) -1

            # The detection_rate is a smaller array, make sure you don't go beyond the end
            detection_index = z_index if z_index < n_redshifts_detection else n_redshifts_detection

            print('You will only save data up to redshift ', maxz, ', i.e. index', z_index)
            save_redshifts        = redshifts[:z_index]
            save_merger_rate      = merger_rate[:,:z_index]
            # save_detection_rate   = detection_rate[:,:detection_index]

        print('save_redshifts', save_redshifts)
        print('shape of save_merger_rate ', np.shape(save_merger_rate))

        #################################################
        # Write the rates as a seperate dataset
        # re-arrange your list of rate parameters
        DCO_to_rate_mask     = COMPAS.DCOmask #save this bool for easy conversion between BSE_Double_Compact_Objects, and CI weights
        rate_data_list       = [DCO['SEED'][DCO_to_rate_mask], DCO_to_rate_mask , save_redshifts,  save_merger_rate]
        #, merger_rate[:,0], save_detection_rate, Average_SF_mass_needed]
        rate_list_names      = ['SEED', 'DCOmask', 'redshifts', 'merger_rate']
        #,'merger_rate_z0', 'detection_rate'+sensitivity, 'Average_SF_mass_needed']
        for i, data in enumerate(rate_data_list):
            print('Adding rate info {} of shape {}'.format(rate_list_names[i], np.shape(data)) )
            # Check if dataset exists, if so, just delete it
            if rate_list_names[i] in h_new[new_rate_group].keys():
                del h_new[new_rate_group][rate_list_names[i]]
            # write rates as a new data set
            dataNew     = h_new[new_rate_group].create_dataset(rate_list_names[i], data=data)

    #Always close your files again ;)
    h_new.close()
    print( ('Done with append_rates :) your new files are here: %s'%(outfilename)).replace('//', '/') )

def delete_rates(path, filename, mu0=0.035, muz=-0.23, sigma0=0.39, sigmaz=0., alpha=0., append_binned_by_z=False):
    """
        Delete the group containing all the rate information from your COMPAS output with weights hdf5 file


        Args:
            path                   --> [string] Path to the COMPAS file that contains the output
            filename               --> [string] Name of the COMPAS file

            mu0                    --> [float]  metallicity dist: expected value at redshift 0
            muz                    --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                 --> [float]  metallicity dist: width at redshhift 0
            sigmaz                 --> [float]  metallicity dist: redshift evolution of width
            alpha                  --> [float]  metallicity dist: skewness (0 = lognormal)
            append_binned_by_z     --> [Bool] to save space, bin rates by redshiftbin and append binned rates

    """
    #################################################
    #Open hdf5 file that we will write on
    print('filename', filename)
    with h5.File(path +'/'+ filename, 'r+') as h_new:
        # The rate info is shaped as BSE_Double_Compact_Objects[COMPAS.DCOmask] , len(redshifts)
        try:
            DCO             = h_new['BSE_Double_Compact_Objects']#
        except:
            DCO             = h_new['DoubleCompactObjects']#

        #################################################
        # Name of the group that has the data stored
        new_rate_group = 'Rates_mu0{}_muz{}_alpha{}_sigma0{}_sigmaz{}_a{}'.format(mu0, muz, alpha, sigma0, sigmaz)
        if append_binned_by_z:
            new_rate_group  = new_rate_group + '_zBinned'

        if new_rate_group not in h_new:
            print(new_rate_group, 'Does not exist, nothing to do here...')
            #Always close your files again ;)
            h_new.close()
            return
        else:
            print('You want to remove this group, %s, from the hdf5 file, removing now..'%(new_rate_group))
            del h_new[new_rate_group]
            #Always close your files again ;)
            h_new.close()
            print('Done with delete_rates :) your files are here: ', path + '/' + filename )

            return
