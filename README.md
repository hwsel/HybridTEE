# HybridTEE: Secure Mobile DNN Execution Using Hybrid Trusted Execution Environment

## HybridTEE Setup Information
The system is developed under Ubuntu 18.04.4, SGX Linux 2.11, and OPTEE 3.8.0. 

#### Prepare for the HybridTEE environment
  1. Copy the `darknet_SGX` folder to `sgx/SampleCode/` directory. Copy all the folders in the `model_info` folder into the `darknet_SGX` directory. Build the `darknet_SGX`.
  
   ```
   cd SampleCode/darknet_SGX
   sudo make SGX_MODE=HW
   ```
  
  2. Build the original optee system.
   
   ```
   mkdir optee_3.8.0
   cd optee_3.8.0
   repo init -u https://github.com/OP-TEE/manifest.git -m rpi3.xml -b 3.8.0
   repo sync -j4 --no-clone-bundle
   cd build
   make -j2 toolchains
   make -j `nproc`
   ```

  3. Copy the `darknetp_optee` folder into the `optee/optee_examples` directory.
   
  4. Copy all the folders in the `model_info` folder into the `optee/out-br/target/root/` directory. 

  5. Update OPTEE source files to make the TA size compatible. The original solution is [here](https://github.com/mofanv/darknetz/issues/2#issuecomment-529763445). More information regarding the issues is [here](https://github.com/mofanv/darknetz/issues/12). The edited files for OPTEE 3.8.0 are in the `optee_config` folder.
      1. In ``optee/optee_os/mk/config.mk``, make the following edits:
      
      ```
      CFG_TEE_TA_LOG_LEVEL ?= 4
      CFG_TEE_TA_MALLOC_DEBUG ?= y
      ```
      2. In ``optee/optee_os/core/arch/arm/plat-rpi3/conf.mk``, make the following edits:

      ```
      CFG_TZDRAM_SIZE ?= 0x04000000
      CFG_TEE_RAM_VA_SIZE ?= 0x00200000
      ```

      3. In ``optee/optee_os/core/arch/arm/include/mm/pgt_cache.h``, comment out or delete these lines:

      ```
      #if CFG_NUM_THREADS < 2
      #define PGT_CACHE_SIZE	4
      #else
      #define PGT_CACHE_SIZE	ROUNDUP(CFG_NUM_THREADS * 2, PGT_NUM_PGT_PER_PAGE)
      #endif
      ```
      
      4. Make the following edits near the commented lines:
      
      ```
      #ifdef CFG_WITH_PAGER
      #if CFG_NUM_THREADS < 2
      #define PGT_CACHE_SIZE	4
      #else
      #define PGT_CACHE_SIZE	ROUNDUP(CFG_NUM_THREADS * 2, PGT_NUM_PGT_PER_PAGE)
      #endif
      #else
      #define PGT_CACHE_SIZE	32
      #endif /*CFG_WITH_PAGER*/
      ```

      4. In ``optee/optee_examples/darknetp_optee/ta/include/user_ta_header_defines.h``, make sure the defined values of the `TA_STACK_SIZE` and the `TA_DATA_SIZE` are:

      ```
      #define TA_STACK_SIZE			(1 * 1024 * 1024)
      #define TA_DATA_SIZE			(60 * 1024 * 1024)
      ```
      
  6. Rebuild the optee system and generate the SD card image.
  
#### Connect the OPTEE and SGX devices in the HybridTEE system
  1. Plug the SD card into the Raspberry pi. Connect an ethernet cable between the pi and the SGX machine. Connect a USB to UART cable between the pi and the SGX machine.

  2. On the system terminal, run the command for the pi UART interface.
   
   ```
   picocom -b 115200 /dev/ttyUSB0
   ```

  3. Now supply power to the Raspberry pi and allow OPTEE to boot. You should see the booting sequence over the UART interface. Then type root and press enter.

  4. Now setup the ethernet connection between the two devices using the following commands in the OPTEE terminal.
      
      ```
      ifconfig eth0 up
      ifconfig eth0 169.254.232.180 netmask 255.255.0.0
      route add default gw 169.254.232.1
      echo "nameserver 8.8.8.8" > /etc/resolv.conf
      ping 169.254.232.175
      ```
      
      NOTE: These IP addressess are a placeholder. You can change them based on your router configuration if needed. Here we set the OPTEE IP address is `169.254.232.180`, and the SGX IP address is `169.254.232.175`.

      If the ping is successful, ethernet has been setup correctly.

   5. Open a second terminal and navigate to `sgx/SampleCode/darknet_SGX` directory.
  
#### Run the HybridTEE system. 
Here is an example to run Darknet19 with the eagle image in the HybridTEE system. Please download the weights file from [here](https://pjreddie.com/darknet/imagenet/) and place it into `model_info/models/darknet/` for both of the SGX and OPTEE folders. 
Assume the total layers that run in the OPTEE are 5 (the first 4 layers + the last layer). The ``-gpp`` flag and the ``-st`` indicate the number of first layers running in the OPTEE. To run other models or the other images, change the cfg, the weights, and the input image files, respectively. Note that the height and the width in the cfg files in the SGX and the OPTEE should be 128 for both (the larger image size might make the OPTEE application fail to run). 
  1. Start the system from the SGX first:
         
   ```
   sudo ./app classifier predict cfg/imagenet1k.data cfg/darknet19.cfg models/darknet/darknet19.weights data/darknet/partial_outputs_trustzone.data data/darknet/filesize.txt data/darknet/tag_trustzone.data data/darknet/tag_size.txt -st 4
   ```
  2. After you see that SGX waiting for data from OPTEE. Run the OPTEE application:
         
   ```
   darknetp classifier predict -pp 0 -gpp 4 cfg/imagenet1k.data cfg/darknet19.cfg models/darknet/darknet19.weights data/darknet/eagle.jpg
   ```   

  3. The entire system starts with remote attestation and run the Darknet inference. Final inference can be seen on the OPTEE console.
   
#### Run Darknet Baseline in the OPTEE Only
  1. Copy the `darknet_baseline` folder into the `optee/optee_examples` directory.
   
  2. Copy all the folders in the `model_info` folder into the `optee/out-br/target/root/` directory. 
  
  3. Rebuild the optee system and generate the SD card image.
  
  4. After boot the OPTEE system with the current SD card image, run the `darknet_baseline`:
  
   ```
   darknet_baseline classifier predict cfg/imagenet1k.data cfg/darknet19.cfg darknet19.weights data/eagle.jpg
   ```

#### Partition Point: Using the Auxiliary DNN Model and SIFT Object Detection Methods
1. Download the official darknet code from [here](https://pjreddie.com/darknet/imagenet/) and follow the instructions there to run the Darknet19 model.

2. Modify the darknet code to save the intermediate outputs of the first 8 layers. Critical files to save the intermediate outputs are in the `darknetp_intermediate` folder. The name format for the intermediate outputs is ``$(image_name)_$(layer_number)_$(channel_number).jpg``. For example, ``giraffe_0_0.jpg`` means the output of the first layer and first channel for the griaffe image. 

3. Now run the `object_detect.py` script with 5 image names as parameters. The test images are in the `testing_images` folder. The `$<DataSize>` evaluated in the paper includes 128, 256, and 448. The `$<ModelName>` evaluated in the paper includes darknet19, vgg16, resnet152, and googlenet. 

  ```
  python3 object_detect.py eagle.jpg dog.jpg cat.jpg horses.jpg giraffe.jpg $<DataSize> $<ModelName>
  ```

4. The test images are in the `testing_images` folder. 

5. This script generates all the intermediate outputs of the first 8 layers for 5 images for both SIFT and YOLO object detection.
