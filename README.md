# HydroGAN

## TODO
* Data
    * ~~Change the subcube sampling procedure to let non-128 multiple cubes to be selected as well~~
        * ~~Current way = (2048/128 - 2)^3 = 2744 different subcubes only~~
        * ~~New way = (2048 - 128*2)^3 = 5.7 billion different subcube combinations~~
    * Change the subcube sampling procedure
        * Currently, the subsampling subsamples all the training samples and stores them.
            * in the self.samples under HydrogenDataset2
        * Change this to quickly sample when the get_samples() function is called.
            * incorporate get_samples into "\_\_getitem\_\_"
            * get_samples should open f itself without reading the whole 2048 cube wholly, sample using the coordinates and f.close()
    * Calculate the total number of different subcubes we can sample from
* VAE
    * Investigate why VAE is not producing any hydrogen masses? (Probably an issue with the decoder part of the VAE)
    * In the decode() of the VAE class, create multiple checkpoints: sum all the values in the **out** variable and plot the evolution of the sum. Compare with the sum of the input subcubes.
* GAN
    * ~~Plotting the evolution of the Generator(noise) subcubes during training~~
    * Check whether the inputs are in the range of (-1,1) -> https://github.com/soumith/ganhacks
    * We don't have any downsampling/upsampling components in both the discriminator and the generator, maybe try those (Downsampling : Average Pooling, Upsampling: PixelShuffle) -> https://github.com/soumith/ganhacks
    * Try Label smoothing? -> https://github.com/soumith/ganhacks
    * Add script to check norms of gradients - if they are over 100 things are screwing up - https://github.com/soumith/ganhacks 
    * while lossD > A: train D, while lossG > B: train G - https://github.com/soumith/ganhacks 
    * MMD-GAN
        * Does a high number of 0's hinder the training procedure?
        * log(a + 1) doesnt transform the distribution to normal!
            * how to overcome this?
            * Can MMD replicate this non-normal distribution?
        * Add f_dec_X_D & f_dec_Y_D to see whether the AE is working correctly
        * Add 3D plots!
* Hybrid Models
    * https://github.com/soumith/ganhacks -> if you cant use DCGANs and no model is stable, use a hybrid model : KL + GAN or VAE + GAN
    * How to use VAE + GAN?
* Validation
    * 1 PDF
    * Log Histograms (compare the real distribution vs. the generated ones)
        * VAE: the output of the decoder
        * GAN: The output of Generator(noise)
    * Power spectrum
    * 3D Plot comparisons



## Data
sample data is in data folder where a .h5 file is put. sample_32.h5 is 32 of randomly sampled cubes with dimensions ?x?x?

