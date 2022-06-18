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

    parser.add_argument('--info', action="store_true", default=False,
                        help='Get basic information concerning the given tartget'
                            )

        
    parser.add_argument('--data',  type=str, default="cube,spec",
                        help='Which kind of data do you want?'+"\n"+\
                             'format: coma separated values "cube,spec,crr"'+"\n"+
                             "NB: cube=e3d files ; spec = spectrum files ; ccd = ccd files "
                            )

    parser.add_argument('--extension',  type=str, default="*",
                        help='Which extension should be file have?'+"\n"+\
                             'format: coma separated values "pdf,fits,png"'
                            )
    parser.add_argument('--source',  type=str, default="local",
                        help='Where are you looking for the files?'+"\n"+\
                             'local: your computer | pharos: online"'
                            )
    
    parser.add_argument('--fileout',  type=str, default=None,
                        help='provide here a path to a file where you want to save the output.'+"\n"+\
                             'if None, the output will be printed'
                            )
    
    parser.add_argument('--quiet', action="store_true", default=False,
                        help='This prints information. Should this be quite?'
                            )
    
    args   = parser.parse_args()
    
    # ---------- #
    #  Target    #    
    # ---------- #
    target = args.infile
    if not args.quiet:
        print(" pysedm find ".center(40,'='))
        print("looking for data associated to: %s"%target)
        print("credit: based on ztfquery")
        
    # ---------- #
    #  Script    #    
    # ---------- #
    if args.info:
        import sys
        print(SEDMQ.get_target_data(target))
        sys.exit()
        
    
    # Downloading options
    datapath = SEDMQ.get_data_path(target, source=args.source,
                                    which=args.data.split(","),
                                    extension=args.extension if not "," in args.extension else args.extension.split(","),
                                    timerange=args.timerange.split(",") if args.timerange is not None else [None,None])
    if args.fileout is None or args.fileout in ["None"]:
        print(" results ".center(40,'='))
        print("\n".join(datapath))
    else:
        with open(args.fileout, 'w') as f:
            for data in datapath:
                f.write(data+"\n")
        
    
