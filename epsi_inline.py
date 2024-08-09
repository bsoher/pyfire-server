import ismrmrd
import os
import sys
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter
import matplotlib.pyplot as plt

# bjs imports
from logging import FileHandler, Formatter

BJS_DEBUG_PATH = "D:\\temp\\debug_fire\\"
LOG_FORMAT = ('%(asctime)s | %(levelname)s | %(message)s')

# Folder for debug output files
# debugFolder = "/tmp/share/debug"
debugFolder = "D:\\temp\\debug_fire"

logger_bjs = logging.getLogger("bjs_log")
logger_bjs.setLevel(logging.DEBUG)

file_handler = FileHandler(os.path.join(BJS_DEBUG_PATH, 'log_epsi_out.txt'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(Formatter(LOG_FORMAT))
logger_bjs.addHandler(file_handler)


# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.setFormatter(Formatter(LOG_FORMAT))
# logger_bjs.addHandler(stdout_handler)


class BlockEpsi:
    """
    Building block object used to create a list of MIDAS processing blocks.

    This object represents the settings and results involved in processing
    data for EPSI.

    In here we also package all the functionality needed to save and recall
    these values to/from an XML node.

    """
    XML_VERSION = "2.0.0"   # allows us to change XML I/O in the future

    def __init__(self, attributes=None):

        # Settings - algorithm flags and parameters

        self.trajectory_filename    = r"g100_r130_sim.dat"      # expand on run
        self.echo_drift_corr        = 1
        self.frequency_drift_corr   = 2
        self.frequency_drift_value  = 0.000
        self.invert_z               = True
        self.swap_lr                = False
        self.echo_output            = 0
        self.echo_average_fix       = True
        self.retain_input_files     = False
        self.plot_echo_positions    = True
        self.nx_resample            = 50
        self.apply_kx_phase_corr    = 1         # deprecated, always done if echoShiftsOpt set

        # multiprocessing settings
        self.single_process     = True          # use multiprocessing or not
        self.nprocess           = 1             # number of cores to use
        self.chunksize          = None          # alignment with other pymidas modules

        # data information
        self.nt                 = None          # these are defaults for Siemens
        self.nx                 = None
        self.ny                 = 50
        self.nz                 = 18
        self.os                 = 2
        self.os_orig            = 2
        self.sw                 = 2500.0
        self.nx_out             = 50
        self.full_traj          = False          # deprecated?
        self.sampling_interval  = 4000

        # dynamically set
        self.do_setup           = True      # setup arrays first off
        
        self.fovx               = 280.0
        self.fovy               = 280.0
        self.fovz               = 180.0
        self.ncha               = 12
        self.n_process_nodes    = 1
        self.data_id_array      = []
        self.nx_resample        = 50
        self.byte_order         = ''
        self.num_phencs         = 1
        self.nd                 = 1
        self.td                 = 1.0
        self.scan_data          = ''
        self.mrdata_fnames      = []

        self.series_label       = 'SI_REF'
        self.pix_spacing_1      = 5.6
        self.pix_spacing_2      = 5.6
        self.pix_spacing_3      = 10.0
        self.mrdata             = None
        self.out_filename       = ''
        self.save_output        = True
        self.channel            = ''
        self.csa_pad_length     = 0
        self.fin_names          = []
        self.fout_names         = []
        self.echo_shifts_slope  = None
        self.echo_shifts        = None
        self.echo_phases        = None
        self.Is_GE              = False
        self.last_zindx        = 0

        # data storage

        self.data_init_met = None
        self.data_init_wat = None
        self.water = []
        self.metab = []

    @property
    def n_channels(self):
        return self.ncha
    @n_channels.setter
    def n_channels(self, value):
        self.ncha = value

    @property
    def nchannels(self):
        return self.ncha
    @nchannels.setter
    def nchannels(self, value):
        self.ncha = value


def process(connection, config, metadata):
    """
    This version is same as the ICE 'Raw' selection. Data is store frame
    by frame to the DICOM database with just raw k-space data in it.

    Input from server is a group of data from each ADC, this method will
    collate the EPI readouts into one group for Metab signals and another
    group for Water signals that comprise one TR acquisition.  Data will
    be saved to one of two arrays that hold all the (nt, nt, nx) encodes
    for one Z phase encode of data. When that array is full, it will be
    sent back from FIRE for storage in the database. And a new array of
    data will be collated.

        csi_se.cpp code re. encoding indices

        PAR - ZPhase 18 - m_adc1.getMDH().setCpar((short) m_sh_3rd_csi_addr[i] + m_sh_3rd_csi_addr_offset);
        LIN - YPhase 50 - m_adc1.getMDH().setClin(SpecVectorSizeshort) m_sh_2nd_csi_addr[i] + m_sh_2nd_csi_addr_offset);
        ECO - 0/1 Water/Metab m_adc1.getMDH().setCeco(0);     WS vs Water Non-Suppressed
        SEG - 100 (50 w/o OS) m_adc1.getMDH().setCseg(ADCctr + +);        EPI RO segment
        m_adc1.getMDH().setFirstScanInSlice(!i && !j);
        m_adc1.getMDH().setLastScanInSlice(i == (m_lN_csi_encodes - 1) && j == (m_sh_csi_weight[i] - 1));
        m_adc1.getMDH().addToEvalInfoMask (MDH_PHASCOR);

    """

    block = BlockEpsi()

    logging.info("Config: \n%s", config)
 
    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
 
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)",
            metadata.encoding[0].trajectory,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.y,
            metadata.encoding[0].encodedSpace.matrixSize.z,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

        block.nz = int(metadata.encoding[0].encodingLimits.kspace_encoding_step_2.maximum - metadata.encoding[0].encodingLimits.kspace_encoding_step_2.minimum) + 1
        block.ny = int(metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum - metadata.encoding[0].encodingLimits.kspace_encoding_step_1.minimum) + 1
        block.nt = mrdhelper.get_userParameterLong_value(metadata, 'SpecVectorSize')
        block.fovx = metadata.encoding[0].encodedSpace.fieldOfView_mm.x
        block.fovy = metadata.encoding[0].encodedSpace.fieldOfView_mm.y
        block.fovz = metadata.encoding[0].encodedSpace.fieldOfView_mm.z

    except:
        logging.info("Improperly formatted metadata or auxiliary variables: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    acq_group = []
    ctr_group = []

    logger_bjs.info("----------------------------------------------------------------------------------------")
    logger_bjs.info("Start EPSI.py run")

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                if block.do_setup:
                    block.ncha, block.nx = item.data.shape

                    dims_init = [block.ncha, 1, 1, block.nt, block.nx]
                    block.data_init_met = np.zeros(dims_init, item.data.dtype)
                    block.data_init_wat = np.zeros(dims_init, item.data.dtype)

                    dims = [block.nz, block.ny, block.nt, block.nx]
                    block.water = []
                    block.metab = []
                    for i in range(block.ncha):
                        block.water.append(np.zeros(dims, item.data.dtype))
                        block.metab.append(np.zeros(dims, item.data.dtype))

                    block.do_setup = False

                flag_ctr_kspace   = item.user_int[0] > 0
                flag_last_epi     = item.user_int[1] > 0
                flag_last_yencode = item.idx.kspace_encode_step_1 == block.ny-1

                if flag_ctr_kspace:             # Center of kspace data
                    ctr_group.append(item)
                    if flag_last_epi:
                        process_init(block, ctr_group, config, metadata)
                        ctr_group = []
                else:                           # Regular kspace acquisition
                    acq_group.append(item)
                    if flag_last_epi:
                        process_group(block, acq_group, config, metadata)

                        if item.idx.contrast == 1 and flag_last_yencode:
                            logger_bjs.info("**** bjs - send_raw() -- zindx = %d, yindx = %d " % (item.idx.kspace_encode_step_2, item.idx.kspace_encode_step_1))
                            images = send_raw(block, acq_group, connection, config, metadata)
                            connection.send_image(images)
                            block.last_zindx += 1

                        acq_group = []

                bob = 10

            elif item is None:
                break
 
            else:
                logging.error("Unsupported data  type %s", type(item).__name__)
 
    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
 
    finally:
        connection.send_close()


def process_init(block, group, config, metadata):
    """ Format data into a [cha RO ave lin seg] array """

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = list(range(block.nt))

    if len(set(indz)) > 1:
        logger_bjs.info("Too many Z encodes in Init data group")
    if len(set(indy)) > 1:
        logger_bjs.info("Too many Y encodes in Init data group")

    if group[0].idx.contrast == 0:
        for acq, it in zip(group, indt):
            block.data_init_met[:, 0, 0, it, :] = acq.data
    else:
        for acq, it in zip(group, indt):
            block.data_init_wat[:, 0, 0, it, :] = acq.data




def process_group(block, group, config, metadata):

    indz = [item.idx.kspace_encode_step_2 for item in group]
    indy = [item.idx.kspace_encode_step_1 for item in group]
    indt = list(range(block.nt))

    index_z = list(set(indz))
    index_y = list(set(indy))

    if len(index_z) > 1:
        logger_bjs.info("Too many Z encodes in TR data group")
    if len(index_y) > 1:
        logger_bjs.info("Too many Y encodes in TR data group")

    for acq, iz, iy, it in zip(group, indz, indy, indt):
        for i in range(block.ncha):
            if group[0].idx.contrast == 0:
                block.metab[i][iz, iy, it, :] = acq.data[i,:]
            else:
                block.water[i][iz, iy, it, :] = acq.data[i,:]
    return


def send_raw(block, group, connection, config, metadata):

    zindx = block.last_zindx
    images = []

    # Set ISMRMRD Meta Attributes
    tmpMeta = ismrmrd.Meta()
    tmpMeta['DataRole'] = 'Spectroscopy'
    tmpMeta['ImageProcessingHistory'] = ['FIRE', 'SPECTRO', 'PYTHON']
    tmpMeta['Keep_image_geometry'] = 1
    tmpMeta['SiemensControl_SpectroData'] = ['bool', 'true']

    # Change dwell time to account for removal of readout oversampling
    dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms

    if dwellTime is None:
        logging.error("Could not find DwellTime_0 in MRD header")
    else:
        logging.info("Found acquisition dwell time from header: " + str(dwellTime * 1000))
        tmpMeta['SiemensDicom_RealDwellTime'] = ['int', str(int(dwellTime * 1000 * 2))]

    xml = tmpMeta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)

    for icha in range(block.ncha):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)

        metab = block.metab[icha][zindx, :,:,:].copy()
        water = block.water[icha][zindx, :,:,:].copy()

        ms = metab.shape
        ws = water.shape
        metab.shape = ms[0], ms[1] * ms[2]
        water.shape = ws[0], ws[1] * ws[2]

        tmpImgMet = ismrmrd.Image.from_array(metab, transpose=False)
        tmpImgWat = ismrmrd.Image.from_array(water, transpose=False)

        # Set the header information
        tmpImgMet.setHead(mrdhelper.update_img_header_from_raw(tmpImgMet.getHead(), group[0].getHead()))
        tmpImgWat.setHead(mrdhelper.update_img_header_from_raw(tmpImgWat.getHead(), group[0].getHead()))

        # 2D spectroscopic imaging
        tmpImgMet.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))
        tmpImgWat.field_of_view = (ctypes.c_float(block.fovx),ctypes.c_float(block.fovy),ctypes.c_float(block.fovz))

        tmpImgMet.image_index = 1
        tmpImgWat.image_index = 1
# bjs    tmpImg.flags = 2 ** 5  # IMAGE_LAST_IN_AVERAGE

        tmpImgMet.attribute_string = xml
        tmpImgWat.attribute_string = xml

        images.append(tmpImgMet)
        images.append(tmpImgWat)

    return images


def process_raw(group, connection, config, metadata):
    # Format data into a [cha RO ave lin seg] array
    nAve = int(metadata.encoding[0].encodingLimits.average.maximum                - metadata.encoding[0].encodingLimits.average.minimum)                + 1
    nLin = int(metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum - metadata.encoding[0].encodingLimits.kspace_encoding_step_1.minimum) + 1
    nSeg = int(metadata.encoding[0].encodingLimits.segment.maximum                - metadata.encoding[0].encodingLimits.segment.minimum)                + 1
    nRO  = mrdhelper.get_userParameterLong_value(metadata, 'SpecVectorSize')

    if nRO is None:
        nRO = int((group[0].data.shape[1] - group[0].discard_pre - group[0].discard_post) / 2)  # 2x readout oversampling
        logging.warning("Could not find SpecVectorSize in header -- using size %d from data", nRO)

    # 2x readout oversampling
    nRO = nRO * 2

    logging.info("MRD header: %d averages, %d lines, %d segments" % (nAve, nLin, nSeg))

    aves = [acquisition.idx.average              for acquisition in group]
    lins = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    segs = [acquisition.idx.segment              for acquisition in group]

    data = np.zeros((group[0].data.shape[0], 
                     nRO,
                     nAve, 
                     nLin, 
                     nSeg), 
                    group[0].data.dtype)

    for acq, ave, lin, seg in zip(group, aves, lins, segs):
        data[:,:,ave,lin,seg] = acq.data[:,acq.discard_pre:(acq.data.shape[1]-acq.discard_post)]

    logging.info("Incoming raw spectroscopy data is shape %s" % (data.shape,))

    isMultiCha = True

    if not isMultiCha:
        # Select coil with the best SNR
        indBestCoil = np.argmax(np.mean(np.abs(data[:,:,0:9,0,0]),axis=(1,2)))
        data = data[np.newaxis,indBestCoil,...]

    # Remove readout oversampling
    data = fft.fft(data, axis=1)
    data = np.delete(data, np.arange(int(data.shape[1]*1/4),int(data.shape[1]*3/4)), axis=1)
    data = fft.ifft( data, axis=1)

    # Match Siemens convention of complex conjugate representation
    data = np.conj(data)

    # Match Siemens data scaling
    data = data * 2**25

    # Image recon for spectroscopic imaging
    if (data.shape[3] > 1) and (data.shape[4] > 1):
        data = fft.fftshift( data, axes=(3, 4))
        data = fft.ifft2(    data, axes=(3, 4))
        data = fft.ifftshift(data, axes=(3, 4))

    # Combine averages
    data = np.mean(data, axis=2, keepdims=True)

    # Collapse into shape [RO lin seg]
    data = np.squeeze(data)

    # Send data back as complex singles
    data = data.astype(np.complex64)

    if isMultiCha:
    # Transpose to shape [SEG LIN COL]
        data = np.transpose(data, (0, 3, 2, 1))
    else:
        data = data.transpose()
    logging.info("Outgoing spectroscopy data is shape %s" % (data.shape,))

    # Create new MRD instance for the processed image
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
    # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)
    tmpImg = ismrmrd.Image.from_array(data, transpose=False)
 
    # Set the header information
    tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), group[0].getHead()))

    if data.ndim > 1:
        # 2D spectroscopic imaging
        tmpImg.field_of_view = (ctypes.c_float(data.shape[2]/data.shape[1]*metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
    else:
        # Single voxel
        tmpImg.field_of_view = (ctypes.c_float(data.shape[0]*metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    tmpImg.image_index   = 1
    tmpImg.flags         = 2**5   # IMAGE_LAST_IN_AVERAGE
 
    logging.info("Outgoing spectroscopy data is field_of_view %s, %s, %s" % (np.double(tmpImg.field_of_view[0]), np.double(tmpImg.field_of_view[1]), np.double(tmpImg.field_of_view[2])))
    logging.info("Outgoing spectroscopy data is matrix_size   %s, %s, %s" % (tmpImg.getHead().matrix_size[0], tmpImg.getHead().matrix_size[1], tmpImg.getHead().matrix_size[2]))

    # Set ISMRMRD Meta Attributes
    tmpMeta = ismrmrd.Meta()
    tmpMeta['DataRole']                            = 'Spectroscopy'
    tmpMeta['ImageProcessingHistory']              = ['FIRE', 'SPECTRO', 'PYTHON']
    tmpMeta['Keep_image_geometry']                 = 1
    tmpMeta['SiemensControl_SpectroData']          = ['bool', 'true']
    #tmpMeta['SiemensControl_Suffix4DataFileName']  = ['string', '-1_1_1_1_1_1']

    # Change dwell time to account for removal of readout oversampling
    dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms

    if dwellTime is None:
        logging.error("Could not find DwellTime_0 in MRD header")
    else:
        logging.info("Found acquisition dwell time from header: " + str(dwellTime*1000))
        tmpMeta['SiemensDicom_RealDwellTime']         = ['int', str(int(dwellTime*1000*2))]
 
    xml = tmpMeta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    tmpImg.attribute_string = xml

    images = [tmpImg]

    roiImg = plot_spectra(tmpImg, connection, config, metadata)
    if roiImg is not None:
        images.append(roiImg)

    return images
 

def process_image(images, connection, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")
 
    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    logging.info("    2D spectroscopic imaging size is %d x %d x %d with %d channels of type %s", images[0].matrix_size[0], images[0].matrix_size[1], images[0].matrix_size[2], images[0].channels, ismrmrd.get_dtype_from_data_type(images[0].data_type))
   
    spectraImgs = process_spectra(images, connection, config, metadata)

    roiImg = plot_spectra(images[0], connection, config, metadata)
    if roiImg is not None:
        spectraImgs.append(roiImg)

    return spectraImgs


def process_spectra(images, connection, config, metadata):   

    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]


    nSpecVectorSize = mrdhelper.get_userParameterLong_value(metadata, 'SpecVectorSize')
    nImgCols = metadata.encoding[0].reconSpace.matrixSize.x
    nImgRows = metadata.encoding[0].reconSpace.matrixSize.y

    spectraOut = [None] * data.shape[0]

    for iImg in range(data.shape[0]):
        # Create new MRD instance for the processed image
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
        # For spectroscopy data, dimensions are: [y x t], i.e. [SEG LIN COL] (PAR would be 3D)
        tmpData = np.squeeze(data[iImg])
        


        # tmpData = tmpData.reshape((nImgRows, nImgCols, nSpecVectorSize))
        tmpData = tmpData.reshape((nImgCols, nImgRows, nSpecVectorSize))
        logging.info("Reshaped back spectroscopy data is shape %s" % (tmpData.shape,))
        
        tmpImg = ismrmrd.Image.from_array(tmpData, transpose=False)
     
        tmpHead = head[iImg]

        tmpHead.matrix_size[0] = nSpecVectorSize
        tmpHead.matrix_size[1] = nImgRows 
        tmpHead.matrix_size[2] = nImgCols

        # Set the header information
        tmpImg.setHead(tmpHead)

        if tmpData.ndim > 1:
            # 2D spectroscopic imaging
            tmpImg.field_of_view = (ctypes.c_float(tmpData.shape[2]/tmpData.shape[1]*metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        else:
            # Single voxel
            tmpImg.field_of_view = (ctypes.c_float(tmpData.shape[0]*metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

        tmpImg.image_index   = iImg
        tmpImg.flags         = 2**5   # IMAGE_LAST_IN_AVERAGE
     
        logging.info("Outgoing spectroscopy data is field_of_view %s, %s, %s" % (np.double(tmpImg.field_of_view[0]), np.double(tmpImg.field_of_view[1]), np.double(tmpImg.field_of_view[2])))
        logging.info("Outgoing spectroscopy data is matrix_size   %s, %s, %s" % (tmpImg.getHead().matrix_size[0], tmpImg.getHead().matrix_size[1], tmpImg.getHead().matrix_size[2]))

        # Set ISMRMRD Meta Attributes
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                            = 'Spectroscopy'
        tmpMeta['ImageProcessingHistory']              = ['FIRE', 'SPECTRO', 'PYTHON']
        tmpMeta['Keep_image_geometry']                 = 1
        tmpMeta['SiemensControl_SpectroData']          = ['bool', 'true']
        #tmpMeta['SiemensControl_Suffix4DataFileName']  = ['string', '-1_1_1_1_1_1']

        # Change dwell time to account for removal of readout oversampling
        dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms

        if dwellTime is None:
            logging.error("Could not find DwellTime_0 in MRD header")
        else:
            logging.info("Found acquisition dwell time from header: " + str(dwellTime*1000))
            tmpMeta['SiemensDicom_RealDwellTime']         = ['int', str(int(dwellTime*1000*2))]
     
        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml

        spectraOut[iImg] = tmpImg

    return spectraOut

def plot_spectra(img, connection, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # For 2D trajectories, create both an ROI vectorized figure and a PNG figure
    if (img.data.shape[1] > 1) or (img.data.shape[2] > 1):
        return None

    # ---------- Send back an MRD image with the spectrum as ROIs ----------
    roiMeta = ismrmrd.Meta()
    roiMeta['SequenceDescriptionAdditional']  = 'SPEC_PLOT'
    roiMeta['WindowCenter']                   = '16384'
    roiMeta['WindowWidth']                    = '32768'
    roiMeta['Keep_image_geometry']            = 1
    roiMeta['InternalSend']                   = ['bool', 'true']

    # Size of blank dummy image
    imgX = 128
    imgY = 128

    # Fraction of image to use
    widthX = 0.9
    heightY = 0.2
    offsetY = 0.4

    # Image coordinates have origin at top left
    y =  fft.fftshift(fft.fft(np.squeeze(img.data), axis=0))
    y = np.abs(y)
    y = ((1-(y/np.max(y)))*heightY+offsetY) * imgY
    x = np.linspace(-widthX/2, widthX/2, len(y))*imgX + imgX/2

    # Plot options
    rgb = (1,0,0)  # Red, green, blue color -- normalized to 1
    thickness  = 1 # Line thickness
    style      = 0 # Line style (0 = solid, 1 = dashed)
    visibility = 1 # Line visibility (0 = false, 1 = true)

    roiMeta['ROI_spectra'] = mrdhelper.create_roi(x, y, rgb, thickness, style, visibility)

    # Additional ROI for x-axis
    xAxis = np.array((-widthX/2, widthX/2))*imgX + imgX/2
    yAxis = (np.array((offsetY,offsetY))+heightY) * imgY
    roiMeta['ROI_axis']    = mrdhelper.create_roi(xAxis, yAxis, (0,0,1), thickness, style, visibility)

    # Blank MRD image
    roiImg = ismrmrd.Image.from_array(np.zeros((imgX, imgY), dtype=np.int16), transpose=False)

    # Set the header information
    tmpHead = img.getHead()
    tmpHead.data_type     = roiImg.data_type
    tmpHead.field_of_view = (ctypes.c_float( imgX), ctypes.c_float( imgY), ctypes.c_float(10))  # Dummy FOV because the spectroscopy FOV isn't appropriate
    tmpHead.matrix_size   = (ctypes.c_ushort(imgX), ctypes.c_ushort(imgY), ctypes.c_ushort(1))
    roiImg.setHead(tmpHead)

    roiImg.image_index        = 1
    roiImg.image_series_index = 1
    roiImg.attribute_string   = roiMeta.serialize()

    return roiImg


def dump_flags(item):
    lines = []
    lines.append("ACQ_IS_DUMMYSCAN_DATA                  = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)))
    lines.append("ACQ_LAST_IN_ENCODE_STEP1               = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)))
    lines.append("ACQ_FIRST_IN_ENCODE_STEP2              = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2)))
    lines.append("ACQ_LAST_IN_ENCODE_STEP2               = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP2)))
    lines.append("ACQ_FIRST_IN_AVERAGE                   = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_AVERAGE)))
    lines.append("ACQ_LAST_IN_AVERAGE                    = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_AVERAGE)))
    lines.append("ACQ_FIRST_IN_SLICE                     = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SLICE)))
    lines.append("ACQ_LAST_IN_SLICE                      = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE)))
    lines.append("ACQ_FIRST_IN_CONTRAST                  = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_CONTRAST)))
    lines.append("ACQ_LAST_IN_CONTRAST                   = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_CONTRAST)))
    lines.append("ACQ_FIRST_IN_PHASE                     = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_PHASE)))
    lines.append("ACQ_LAST_IN_PHASE                      = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_PHASE)))
    lines.append("ACQ_FIRST_IN_REPETITION                = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_REPETITION)))
    lines.append("ACQ_LAST_IN_REPETITION                 = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION)))
    lines.append("ACQ_FIRST_IN_SET                       = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SET)))
    lines.append("ACQ_LAST_IN_SET                        = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_SET)))
    lines.append("ACQ_FIRST_IN_SEGMENT                   = " + str(item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SEGMENT)))
    lines.append("ACQ_LAST_IN_SEGMENT                    = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_SEGMENT)))
    lines.append("ACQ_IS_NOISE_MEASUREMENT               = " + str(item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)))
    lines.append("ACQ_IS_PARALLEL_CALIBRATION            = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)))
    lines.append("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING= " + str(item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)))
    lines.append("ACQ_IS_REVERSE                         = " + str(item.is_flag_set(ismrmrd.ACQ_IS_REVERSE)))
    lines.append("ACQ_IS_NAVIGATION_DATA                 = " + str(item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)))
    lines.append("ACQ_IS_PHASECORR_DATA                  = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)))
    lines.append("ACQ_LAST_IN_MEASUREMENT                = " + str(item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT)))
    lines.append("ACQ_IS_HPFEEDBACK_DATA                 = " + str(item.is_flag_set(ismrmrd.ACQ_IS_HPFEEDBACK_DATA)))
    lines.append("ACQ_IS_DUMMYSCAN_DATA                  = " + str(item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA)))
    lines.append("ACQ_IS_RTFEEDBACK_DATA                 = " + str(item.is_flag_set(ismrmrd.ACQ_IS_RTFEEDBACK_DATA)))
    lines.append("ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA  = " + str(item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA)))
    lines.append("ACQ_IS_PHASE_STABILIZATION_REFERENCE   = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE)))
    lines.append("ACQ_IS_PHASE_STABILIZATION             = " + str(item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION)))
    lines.append("ACQ_COMPRESSION1                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION1)))
    lines.append("ACQ_COMPRESSION2                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION2)))
    lines.append("ACQ_COMPRESSION3                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION3)))
    lines.append("ACQ_COMPRESSION4                       = " + str(item.is_flag_set(ismrmrd.ACQ_COMPRESSION4)))
    lines.append("ACQ_USER1                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER1)))
    lines.append("ACQ_USER2                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER2)))
    lines.append("ACQ_USER3                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER3)))
    lines.append("ACQ_USER4                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER4)))
    lines.append("ACQ_USER5                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER5)))
    lines.append("ACQ_USER6                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER6)))
    lines.append("ACQ_USER7                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER7)))
    lines.append("ACQ_USER8                              = " + str(item.is_flag_set(ismrmrd.ACQ_USER8)))

    lines = "\n".join(lines)
    print(lines)

def dump_active_flags(item, prnt=False):
    lines = []

    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1): lines.append("ACQ_IS_DUMMYSCAN_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1):	lines.append("ACQ_LAST_IN_ENCODE_STEP1               = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP2):	lines.append("ACQ_FIRST_IN_ENCODE_STEP2              = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_ENCODE_STEP2):	lines.append("ACQ_LAST_IN_ENCODE_STEP2               = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_AVERAGE):		lines.append("ACQ_FIRST_IN_AVERAGE                   = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_AVERAGE):		lines.append("ACQ_LAST_IN_AVERAGE                    = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SLICE):		lines.append("ACQ_FIRST_IN_SLICE                     = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):		    lines.append("ACQ_LAST_IN_SLICE                      = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_CONTRAST):		lines.append("ACQ_FIRST_IN_CONTRAST                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_CONTRAST):		lines.append("ACQ_LAST_IN_CONTRAST                   = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_PHASE):		lines.append("ACQ_FIRST_IN_PHASE                     = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_PHASE):		    lines.append("ACQ_LAST_IN_PHASE                      = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_REPETITION):	lines.append("ACQ_FIRST_IN_REPETITION                = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):	lines.append("ACQ_LAST_IN_REPETITION                 = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SET):		    lines.append("ACQ_FIRST_IN_SET                       = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SET):			lines.append("ACQ_LAST_IN_SET                        = True")
    if item.is_flag_set(ismrmrd.ACQ_FIRST_IN_SEGMENT):		lines.append("ACQ_FIRST_IN_SEGMENT                   = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SEGMENT):		lines.append("ACQ_LAST_IN_SEGMENT                    = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):	lines.append("ACQ_IS_NOISE_MEASUREMENT               = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION): lines.append("ACQ_IS_PARALLEL_CALIBRATION          = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING): lines.append("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING= True")
    if item.is_flag_set(ismrmrd.ACQ_IS_REVERSE):			lines.append("ACQ_IS_REVERSE                         = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA):	lines.append("ACQ_IS_NAVIGATION_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):		lines.append("ACQ_IS_PHASECORR_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):	lines.append("ACQ_LAST_IN_MEASUREMENT                = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_HPFEEDBACK_DATA):	lines.append("ACQ_IS_HPFEEDBACK_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):		lines.append("ACQ_IS_DUMMYSCAN_DATA                  = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_RTFEEDBACK_DATA):	lines.append("ACQ_IS_RTFEEDBACK_DATA                 = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA): lines.append("ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE):lines.append("ACQ_IS_PHASE_STABILIZATION_REFERENCE   = True")
    if item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION):lines.append("ACQ_IS_PHASE_STABILIZATION             = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION1):		    lines.append("ACQ_COMPRESSION1                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION2):		    lines.append("ACQ_COMPRESSION2                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION3):		    lines.append("ACQ_COMPRESSION3                       = True")
    if item.is_flag_set(ismrmrd.ACQ_COMPRESSION4):		    lines.append("ACQ_COMPRESSION4                       = True")
    if item.is_flag_set(ismrmrd.ACQ_USER1):			        lines.append("ACQ_USER1                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER2):			        lines.append("ACQ_USER2                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER3):		    	    lines.append("ACQ_USER3                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER4):		    	    lines.append("ACQ_USER4                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER5):		    	    lines.append("ACQ_USER5                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER6):		    	    lines.append("ACQ_USER6                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER7):			        lines.append("ACQ_USER7                              = True")
    if item.is_flag_set(ismrmrd.ACQ_USER8):			        lines.append("ACQ_USER8                              = True")

    if lines == []:
        lines = 'No active flags.'
    else:
        lines = "\n".join(lines)

    if prnt == True:
        print(lines)
    else:
        return lines

