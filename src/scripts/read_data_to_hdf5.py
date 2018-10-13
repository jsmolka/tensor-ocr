from model import nistdataorganizer as ndo

# C:\Users\Tom\fast\NIST\by_class
# D:\Users\Tom\Documents\NIST\test

nist_input = input('Please input the NIST directory from where the data will be read: ')
nist_output = input('Please input the output directory for the HDF5 files: ')

org = ndo.DataOrganizer(nist_input, nist_output)
print(org.nist_input)
print(org.nist_output)
org.process_data()
