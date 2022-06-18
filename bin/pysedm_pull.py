#! /usr/bin/env python
#

""" Module to pull pysedm data """


try:
    import ztfquery
    from ztfquery import sedm
except ImportError:
    raise ImportError("Please install ztfquery (v>1.6.0). pip install ztfquery")

# ================= #
#    Main object    #
# ================= #
SEDMQ = sedm.SEDMQuery()


# ======================== #
#                          #
#         MAIN             #
#                          #
# ======================== #
if  __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(
             description=""" pull SEDM data from pharos to your computer""",
             formatter_class=argparse.RawTextHelpFormatter)

    # - Infile - #
    parser.add_argument('infile', type=str, default=None,
                        help='Target Name')

    # ---------- #
    #  Options   #    
    # ---------- #
    parser.add_argument('--timerange',  type=str, default="2018-08-01,now",
                        help='You only one sedm data taken during that time range.'+"\n"+\
                             'format: coma separated values "a,b" \n Could be: '+"\n"+\
                             '- 1 time means starting point (here after the 12th of Sept 2018):  --timerange 2018-09-12  '+"\n"+\
                             '- 2 times means boundaries  (here between 12th and 15th of Sept 2018):  --timerange 2018-09-12,2018-09-15'+"\n"+\
                             '- for only upper limit  (here before 15th of Sept 2018):  --timerange None,2018-09-15'
                            )

    parser.add_argument('--update',  action="store_true", default=False,
                        help='Update the sedm local database of what have been observed (what-files)'
                            )

    parser.add_argument('--data',  type=str, default="cube,spec",
                        help='Which kind of data do you want?'+"\n"+\
                             'format: coma separated values "cube,spec,crr"'+"\n"+
                             "NB: cube=e3d files ; spec = spectrum files ; ccd = ccd files "
                            )

    parser.add_argument('--fluxcal',  action="store_true", default=False,
                        help='Do you want to download the fluxcal files of the nights with your target?'
                            )
        
    parser.add_argument('--output',  type=str, default="default",
                        help='Where should the files be downloaded? (here provide the full path of the directory)'+"\n"+\
                             "by default, it will be in your $ZTFDATA/SEDM/redux/DATES directories following pharos' structure"
                            )

    parser.add_argument('--extension',  type=str, default="*",
                        help='Which extension should be file have?'+"\n"+\
                             'format: coma separated values "pdf,fits,png"'
                            )
    
    parser.add_argument('--overwrite',  action="store_true", default=False,
                        help='If the file you want to download already exists, shall this overwrite it?'
                            )
    
    parser.add_argument('--nodl',  action="store_true", default=False,
                        help='Use this option to not download (only see what should be downloaded where)'
                            )
    
    parser.add_argument('--nprocess',  type=int, default=None,
                        help='How many parallel downloading ?'
                            )
    
    parser.add_argument('--quiet', action="store_true", default=False,
                        help='This prints information. Should this be quite?'
                            )
    
    args   = parser.parse_args()
    
    # ---------- #
    #  Target    #    
    # ---------- #
    if args.update:
        if not args.quiet:
            print("Updating sedmdata")
        SEDMQ.update_sedmdata()
        

    
    target = args.infile
    if not args.quiet:
        print(" pysedm pull ".center(40,'='))
        print("Downloading data for target: %s"%target)
        print("credit: based on ztfquery")
        
    # ---------- #
    #  Script    #    
    # ---------- #
    
    # Downloading options
    dl_options = dict(nodl=args.nodl,download_dir=args.output,
                     nprocess=args.nprocess, verbose=args.quiet,
                     overwrite=args.overwrite)

    # Actual Download function for pysedm files associated to the target
    url, localdl = SEDMQ.download_target_data(target, which=args.data.split(","),
                                   timerange=args.timerange.split(",") if args.timerange is not None else [None,None],
                                   extension=args.extension if not "," in args.extension else args.extension.split(","),
                                   **dl_options
                                )
    # Downloading associated fluxcalibration files if requested
    if args.fluxcal:
        url_fluxcal,localdl_fluxcal = [], []
        target_nights = np.unique(SEDMQ.get_target_data(target)["night"])
        if not args.quiet:
            print("info: downloading fluxcalibration files for night(s): "+", ".join(target_nights))
            
        for target_night in target_nights:
            url_fluxcal_,localdl_fluxcal_ = SEDMQ.download_night_fluxcal(target_night, **dl_options)
            url_fluxcal    +=url_fluxcal_
            localdl_fluxcal+=localdl_fluxcal_
                                           
        url    += url_fluxcal
        localdl+= localdl_fluxcal

    # ---------- #
    #  Outputs   #    
    # ---------- #
    if args.nodl:
        if not args.quiet:
            print(" no download option (--nodl) ".center(40,'='))
            print(" Data to be downloaded ".center(40,'-'))
            print("\n".join(url))
            print(" Download location ".center(40,'-'))
            print("\n".join(localdl))
            

    


    
