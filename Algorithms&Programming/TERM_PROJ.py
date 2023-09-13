# Import libraries
import cv2
import docopt
import numpy as np


class Project():
    # Initialisation of our work field: reading the characteristics of input image: width, height, channels
    def __init__(self, im):
        self.image = im
        self.height, self.width, self.non_bi_channels = im.shape
        self.size = self.width * self.height

        # The schemes of masking - OR and AND, NB: we need to remove the first used bit!
        self.maskONEValues = [1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7]
        self.maskONE = self.maskONEValues.pop(0)
        self.maskZEROValues = [255 - (1 << i) for i in range(8)]
        self.maskZERO = self.maskZEROValues.pop(0)

        # Current position
        self.current_width = 0
        self.current_height = 0
        self.current_channel = 0

    # The function that literally inserts bits into input image
    def climb_inside(self, bits):

        for c in bits:  # Iterate over all bits: look inside 3D matrix
            val = list(self.image[self.current_height, self.current_width])
            # Fix in the image the info about the bit: if it is set - change, if not - remain the bit from the input pic
            # Depending on the status we chose the certain mask
            if int(c):
                val[self.current_channel] = int(val[self.current_channel]) | self.maskONE
            else:
                val[self.current_channel] = int(val[self.current_channel]) & self.maskZERO

            # Update image
            self.image[self.current_height, self.current_width] = tuple(val)

            # The function described below - move pointer to the next space
            self.next_step()

    # To check if it is appropriate to move to the next step
    def next_step(self):
        # Firstly - check if we are still inside the workflow (3 checks for each parameter)
        if self.current_channel == self.non_bi_channels - 1:
            self.current_channel = 0
            if self.current_width == self.width - 1:
                self.current_width = 0
                if self.current_height == self.height - 1:
                    self.current_height = 0
                    # Check if we are on the final step
                    if self.maskONE == 128:
                        raise SteganographyException("No available slot remaining (image filled)")
                    else:  # else go to next bitmask
                        self.maskONE = self.maskONEValues.pop(0)
                        self.maskZERO = self.maskZEROValues.pop(0)
                else:
                    self.current_height += 1
            else:
                self.current_width += 1
        else:
            self.current_channel += 1

    # Catch the certain bit for our analysis
    def read_bit(self):  # Read a single bit int the image
        val = self.image[self.current_height, self.current_width][self.current_channel]
        val = int(val) & self.maskONE
        # move pointer to next location after reading in bit
        self.next_step()

        # Check if current bitmask and val have same set bit
        if val > 0:
            return "1"
        else:
            return "0"

    def read_byte(self):
        return self.read_bits(8)

    # Function to read nb number of bits with check for mask
    def read_bits(self, nb):
        bits = ""
        for i in range(nb):
            bits += self.read_bit()
        return bits

    # Function to generate the byte value of an int and return it
    def byte_value(self, val):
        return self.binary_value(val, 8)

    # Function that returns the binary value of an int as a byte
    def binary_value(self, val, bitsize):
        # Extract binary equivalent
        binval = bin(val)[2:]

        # Another check if we are still on the board
        if len(binval) > bitsize:
            raise SteganographyException("Binary value larger than the expected size, FAIL")

        # Making it 8-bit by prefixing with zeroes
        while len(binval) < bitsize:
            binval = "0" + binval

        return binval

    '''
        The main body, included all parts. 
        Chooses one depending on users preference
        '''

    def encode_text(self, txt):
        length = len(txt)
        bin_length = self.binary_value(length, 16)  # Generates 4 byte binary value of the length of the secret msg
        self.climb_inside(bin_length)  # Put text length coded on 4 bytes
        for char in txt:  # And put all the chars
            c = ord(char)
            self.climb_inside(self.byte_value(c))  # update the image 
        return self.image

    def decode_text(self):
        ls = self.read_bits(16)  # Read the text size in bytes
        l = int(ls, 2)  # Returns decimal value ls
        i = 0
        hidden_text = ""
        while i < l:  # Read all bytes of the text
            tmp = self.read_byte()  # So one byte
            i += 1
            hidden_text += chr(int(tmp, 2))  # Every chars concatenated to str
        return hidden_text


flag = 0

print("\n")
while flag != 3:

    Message = "Which operation would you like to perform?\n1.Encode Text\n2.Decode Text\n3.Exit Program\n"

    # Print to console in center-display format
    print('\n'.join('{:^40}'.format(s) for s in Message.split('\n')))

    flag = int(input())
    if flag == 3:
        break

    if flag == 1:
        print("Please describe the path to the image: ")
        work_dir = input()

        # Create object of class
        obj = Project(cv2.imread(work_dir))

        print("\nWhich message we should hide? ")
        msg = input()

        # Invoke encode_text() function
        print("\nCreating encrypted image.")
        encrypted_img = obj.encode_text(msg)

        print("\nPut the destination and the name of file with secret info: ")
        path = input()

        # Write into destination
        print("\nSaving image in destination.")
        cv2.imwrite(path, encrypted_img)

        print("Encryption complete.")
        print("The encrypted file is available at", path, "\n")

    elif flag == 2:
        print("Enter working directory of source image: ")
        work_dir = input()
        img = cv2.imread(work_dir)
        obj = Project(img)

        print("\nText message obtained from decrypting image is:", obj.decode_text(), "\n")

print("Thank you.")
