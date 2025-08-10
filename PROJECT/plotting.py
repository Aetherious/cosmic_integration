def plot_rates(save_dir, formation_rate, merger_rate, detection_rate, redshifts, chirp_masses, show_plot = False, mu0=0.035, muz=-0.23, sigma0=0.39, sigmaz=0., alpha=0,aSF = 0.02,  bSF = 1.48, cSF = 4.45, dSF = 5.91):
    """
        Show a summary plot of the results, it also returns the summaries that it computes

        Args:
            save_dir                  --> [string] path where you would like to save your plot
            formation_rate            --> [2D float array] Formation rate for each binary at each redshift in 1/yr/Gpc^3
            merger_rate               --> [2D float array] Merger rate for each binary at each redshift in 1/yr/Gpc^3
            detection_rate            --> [2D float array] Detection rate for each binary at each redshift in 1/yr
            redshifts                 --> [list of floats] List of redshifts
            chirp_masses              --> [list of floats] Chrirp masses of merging DCO's

            show_plot                 --> [bool] Bool whether to show plot or not
            mu0                       --> [float]  metallicity dist: expected value at redshift 0
            muz                       --> [float]  metallicity dist: redshift evolution of expected value
            sigma0                    --> [float]  metallicity dist: width at redshhift 0
            sigmaz                    --> [float]  metallicity dist: redshift evolution of width
            alpha                     --> [float]  metallicity dist: skewness (0 = lognormal)

        Returns:
            matplotlib figure

    """
    # sum things up across binaries
    total_formation_rate = np.sum(formation_rate, axis=0)
    total_merger_rate = np.sum(merger_rate, axis=0)
    total_detection_rate = np.sum(detection_rate, axis=0)
    
    # and across redshifts
    cumulative_detection_rate = np.cumsum(total_detection_rate)
    detection_rate_by_binary = np.sum(detection_rate, axis=1)

    ###########################
    #Start plotting

    # set some constants for the plots
    plt.rc('font', family='serif')
    fs = 20
    lw = 3

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    axes[0,0].plot(redshifts, total_formation_rate, lw=lw)
    axes[0,0].set_xlabel('Redshift', fontsize=fs)
    axes[0,0].set_ylabel(r'Formation rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=fs)

    axes[0,1].plot(redshifts, total_merger_rate, lw=lw)
    axes[0,1].set_xlabel('Redshift', fontsize=fs)
    axes[0,1].set_ylabel(r'Merger rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}Gpc^3 \mathrm{d}yr}]$', fontsize=fs)

    axes[1,0].plot(redshifts[:len(cumulative_detection_rate)], cumulative_detection_rate, lw=lw)
    axes[1,0].set_xlabel('Redshift', fontsize=fs)
    axes[1,0].set_ylabel(r'Cumulative detection rate $[\rm \frac{\mathrm{d}N}{\mathrm{d}yr}]$', fontsize=fs)

    axes[1,1].hist(chirp_masses, weights=detection_rate_by_binary, bins=25, range=(0, 50))
    axes[1,1].set_xlabel(r'Chirp mass, $\mathcal{M}_c$', fontsize=fs)
    axes[1,1].set_ylabel(r'Mass distrbution of detections $[\rm \frac{\mathrm{d}N}{\mathrm{d}\mathcal{M}_c \mathrm{d}yr}]$', fontsize=fs)

    #########################
    #Plotvalues

    # Add text upper left corner
    axes[0,0].text(0.05,0.8, "mu0=%s \nmuz=%s \nsigma0=%s \nsigmaz=%s \nalpha=%s"%(mu0,muz,sigma0,sigmaz,alpha), transform=axes[0,0].transAxes, size = fs) 

    for ax in axes.flatten():
        ax.tick_params(labelsize=0.9*fs)

    print("Plotting!")
    # Save and show :)
    plt.savefig(save_dir +'Rate_Info'+"mu0%s_muz%s_alpha%s_sigma0%s_sigmaz%s_a%s_b%s_c%s_d%s"%(mu0,muz,alpha,sigma0,sigmaz,aSF, bSF,cSF,dSF)+'.png', bbox_inches='tight') 
    if show_plot:
        plt.show()
    else:
        plt.close()
