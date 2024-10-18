# mount the image in a loopback device
mkdir mount
sudo mount -o loop,offset=1048576 gem5-resources/src/x86-ubuntu-gpu-ml/disk-image/x86-ubuntu-gpu-ml mount

# copy all files in the DNNMARK folder to the image
# remove the DNNMark folder if it already exists
sudo rm -r mount/home/gem5/DNNMark
sudo cp -r gem5-resources/src/gpu/DNNMark mount/home/gem5

# unmount the image
sudo umount mount
