### Python script to use SNAP as InSAR processor compatible with StaMPS PSI processing
# Author Mohammad Hossein Moazezian
# Date: 25/10/2022
# Version: 1.0

# Step 1 : preparing slaves in folder structure
# Step 2 : TOPSAR Splitting (Assembling) and Apply Orbit
# Step 3 : Coregistration and Interferogram generation
# Step 4 : StaMPS export

# Added option for CACHE and CPU specification by user
# Planned support for DEM selection and ORBIT type selection 


import os
from pathlib2 import Path
import sys
import glob
import subprocess
import shlex
import time
inputfile = sys.argv[1]

bar_message='\n#####################################################################\n'



MONTH = ['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sep' , 'Oct' , 'Nov' , 'Dec']
inputfile = '/media/data/Project_subsidence/project.conf'
# Getting configuration variables from inputfile
try:
        in_file = open(inputfile, 'r')

        for line in in_file.readlines():
                if "PROJECTFOLDER" in line:
                        PROJECT = line.split('=')[1].strip()
                        print PROJECT
                if "IW1" in line:
                        IW = line.split('=')[1].strip()
                        print IW
                if "MASTER" in line:
                        MASTER = line.split('=')[1].strip()
                        print MASTER
                if "GRAPHSFOLDER" in line:
                        GRAPH = line.split('=')[1].strip()
                        print GRAPH
                if "GPTBIN_PATH" in line:
                        GPT = line.split('=')[1].strip()
                        print GPT
		if "LONMIN" in line:
			LONMIN = line.split('=')[1].strip()
                if "LATMIN" in line:
                        LATMIN = line.split('=')[1].strip()
                if "LONMAX" in line:
                        LONMAX = line.split('=')[1].strip()
                if "LATMAX" in line:
                        LATMAX = line.split('=')[1].strip()
		if "CACHE" in line:
			CACHE = line.split('=')[1].strip()
		if "CPU" in line:
			CPU = line.split('=')[1].strip()
finally:
        in_file.close()

polygon = 'POLYGON (('+LONMIN+' '+LATMIN+','+LONMAX+' '+LATMIN+','+LONMAX+' '+LATMAX+','+LONMIN+' '+LATMAX+','+LONMIN+' '+LATMIN+'))'
print polygon
######################################################################################
## TOPSAR Coregistration and Interferogram formation ##
######################################################################################
raw_data_folder = '/media/data/Project_subsidence/raw_data'
outputifgfolder = PROJECT
logfolder = '/media/data/Project_subsidence'+'/logs'
if not os.path.exists(outputifgfolder):
                os.makedirs(outputifgfolder)
if not os.path.exists(logfolder):
                os.makedirs(logfolder)

outlog=logfolder+'/coreg_ifg_proc_stdout.log'

graphxml=GRAPH+'/interferogram.xml'
print(graphxml)

graph2run=GRAPH+'/interferogram_2run.xml'

out_file = open(outlog, 'a')
err_file=out_file

print(bar_message)
out_file.write(bar_message)
message='## Coregistration and Interferogram computation started:\n'
print(message)
out_file.write(message)
print(bar_message)
out_file.write(bar_message)
k=0
filenames = glob.glob("/media/data/Project_subsidence/raw_data/*.zip")


for i in range(73 ,len(filenames)-1):
    master = filenames[i]
    date_master = filenames[i][57:65]
    year_master = date_master[0:4]
    month_master = MONTH[int(date_master[4:6]) - 1]
    day_master = date_master[6:8]

    for j in range(i+1 , len(filenames)):
        slave = filenames[j]
        date_slave = filenames[j][57:65]
        year_slave = date_slave[0:4]
        month_slave = MONTH[int(date_slave[4:6]) - 1]
        day_slave = date_slave[6:8]
        print(i,j)
        k=k+1

        message = '['+str(k)+'] Processing ifg file : \n' + str(date_master)+'  '+ 'and'+ '  ' + str(date_slave)
        print message
        out_file.write(message)
        outputname = date_master+'_'+date_slave+'_'+IW
        outputname2 = day_master + month_master + year_master + '_' + day_slave + month_slave + year_slave

        with open(graphxml, 'r') as file :
           filedata = file.read()
        # Replace the target string
        filedata = filedata.replace('MASTER',master)
        filedata = filedata.replace('SLAVE', slave)
        filedata = filedata.replace('OUTPUTIFGFOLDER', outputifgfolder)
        filedata = filedata.replace('OUTPUTFILE',outputname)
        filedata = filedata.replace('NAME', outputname2)
        filedata = filedata.replace('POLYGON',polygon)
        # Write the file out again
        with open(graph2run, 'w') as file:
           file.write(filedata)
        args = [ GPT, graph2run, '-c', CACHE, '-q', CPU]
        # Launch the processing
        process = subprocess.Popen(args, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        timeStarted = time.time()
        stdout = process.communicate()[0]
        print 'SNAP STDOUT:{}'.format(stdout)
        timeDelta = time.time() - timeStarted                     # Get execution time.
        print('['+str(k)+'] Finished process in '+str(timeDelta)+' seconds.')
        out_file.write('['+str(k)+'] Finished process in '+str(timeDelta)+' seconds.\n')
        if process.returncode != 0 :
            message='Error computing with coregistration and interferogram generation of splitted'+ str(date_master)+ 'and' + str(date_slave)
            err_file.write(message+'\n')
        else:
            message='Coregistration and Interferogram computation for data '+str(date_master)+ 'and' + str(date_slave) +' successfully completed.\n'
            print(message)
            out_file.write(message)
        print(bar_message)
        out_file.write(bar_message)
out_file.close()




