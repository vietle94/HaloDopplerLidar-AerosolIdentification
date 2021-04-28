import classifier
import argparse
import glob

parser = argparse.ArgumentParser(description="Description for arguments")
parser.add_argument("file_location", help="File location", type=str)
parser.add_argument("out_directory", help="Output directory", type=str)
parser.add_argument("-xr", "--XRdevice", help="If data is from Uto-32XR",
                    action='store_true')
parser.add_argument("-d", "--diagnostic", help="Output a diagnostic image \
                    for classification algorithm",
                    action='store_true')
argument = parser.parse_args()

data_list = glob.glob(argument.file_location + "/*.nc")

count = 0
for file in data_list:
    print('Working on ', file)
    classifier.classification_algorithm(file,
                                        argument.out_directory,
                                        diagnostic=argument.diagnostic,
                                        xr_data=argument.XRdevice)
    print('Classified successfully ', file)
    count += 1
print(f'Finished {count} files')
