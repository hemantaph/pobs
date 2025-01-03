def meta_dict(num_images=2):
    
    meta_dict_all = {
            "posterior": {
                "scaling_param": {
                    "mass_1": "mass_scaler",
                    "mass_2": "mass_scaler",
                    "theta_jn": "inclination_scaler",
                    "dl": 'distance_scaler',
                },
            },
            "posterior_sky": {
                "scaling_param": {
                    "ra": "ra_scaler",
                    "dec": None,
                },
            },
            "astro_lensed": {
                "scaling_param": {
                    "mass_1": "mass_scaler",
                    "mass_2": "mass_scaler",
                    "theta_jn": "inclination_scaler",
                    "dl_1": 'distance_scaler',
                    "dl_2": 'distance_scaler',
                    "dt_12": 'time_scaler',
                },
            },
            "astro_lensed_sky": {
                "scaling_param": {
                    "ra": "ra_scaler",
                    "dec": None,
                },
            },
            "posterior_combined": {
                "scaling_param": {
                    "mass_1": "mass_scaler",
                    "mass_2": "mass_scaler",
                    "theta_jn": "inclination_scaler",
                    "dl_1": 'distance_scaler',
                    "dl_2": 'distance_scaler',
                },
            },
            "posterior_combined_sky": {
                "scaling_param": {
                    "ra": "ra_scaler",
                    "dec": None,
                },
            },
            "astro_unlensed": {
                "scaling_param": {
                    "mass_1": "mass_scaler",
                    "mass_2": "mass_scaler",
                    "theta_jn": "inclination_scaler",
                    "dl": 'distance_scaler',
                },
            },
            "astro_unlensed_sky": {
                "scaling_param": {
                    "ra": "ra_scaler",
                    "dec": None,
                },
            },
            "astro_unlensed_time": {
                "scaling_param": {
                    "dt": 'time_scaler',
                },
            },
        }

    if num_images == 2:
            pass

    return meta_dict_all
