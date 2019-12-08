import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utilities import projection
from preprocessing import erase_points
from segmentation import line_horizontal_projection, word_vertical_projection

def binarize(word_img):

    _, binary_img = cv.threshold(word_img, 127, 255, cv.THRESH_BINARY)
    # _, binary_img = cv.threshold(word_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    return binary_img // 255


def baseline_detection(word_img):
    '''Get baseline index of a given word'''

    HP = projection(word_img, 'horizontal')
    peak = np.amax(HP)

    # Array of indices of max element
    baseline_idx = np.where(HP == peak)[0]

    # Get first or last index
    upper_base = baseline_idx[0]
    lower_base = baseline_idx[-1]
    thickness = abs(lower_base - upper_base) + 1
    
    return upper_base, lower_base, thickness


def horizontal_transitions(word_img, baseline_idx):
    
    max_transitions = 0
    max_transitions_idx = baseline_idx
    line_idx = baseline_idx

    # new temp image with no dots above baseline
    tmp_word = word_img.copy()
    erase_points(tmp_word, baseline_idx)
    
    while line_idx >= 0:
        current_transitions = 0
        flag = 0

        horizontal_line = tmp_word[line_idx, :]
        for pixel in reversed(horizontal_line):

            if pixel == 1 and flag == 0:
                current_transitions += 1
                flag = 1
            elif pixel == 0 and flag == 1:
                current_transitions += 1
                flag = 0
                
        if current_transitions >= max_transitions:
            max_transitions = current_transitions
            max_transitions_idx = line_idx

        line_idx -= 1

    return max_transitions_idx


def vertical_transitions(word_img, cut):
    
    transitions = 0

    vertical_line = word_img[:, cut]

    flag = 0
    for pixel in vertical_line:

        if pixel == 1 and flag == 0:
            transitions += 1
            flag = 1
        elif pixel == 0 and flag == 1:
            transitions += 1
            flag = 0

    return transitions


def cut_points(word_img, VP, MFV, MTI):
      
    # flag to know the start of the word
    f = 0

    flag = 0
    (h, w) = word_img.shape
    i = w-1
    separation_regions = []

    # loop over the width of the image from right to left
    while i >= 0:

        pixel = word_img[MTI, i]
        
        if pixel == 1 and f == 0:
            f = 1
            flag = 1

        if f == 1:

            # Get start and end of separation region (both are black pixels <----)
            if pixel == 0 and flag == 1:
                start = i+1
                flag = 0
            elif pixel == 1 and flag == 0:
                end = i         # end maybe = i not i+1
                flag = 1

                mid = (start + end) // 2

                left_zero = -1
                left_MFV = -1
                right_zero = -1
                right_MFV = -1
                # threshold for MFV
                T = 1

                j = mid - 1
                # loop from mid to end to get nearest VP = 0 and VP = MFV
                while j >= end:
                    
                    if VP[j] == 0 and left_zero == -1:
                        left_zero = j
                    if MFV <= VP[j] <= MFV + T and left_MFV == -1:
                        left_MFV = j

                    if left_zero != -1 and left_MFV != -1:
                        break

                    j -= 1

                j = mid + 1
                # loop from mid to start to get nearest VP = 0 and VP = MFV
                while j <= start:

                    if VP[j] == 0 and right_zero == -1:
                        right_zero = j
                    if MFV <= VP[j] <= MFV + T and right_MFV == -1:
                        right_MFV = j

                    if right_zero != -1 and right_MFV != -1:
                        break

                    j += 1

                # Check for VP = 0 first
                if VP[mid] == 0:
                    cut_index = mid
                elif left_zero != -1 and right_zero != -1:
                    
                    if abs(left_zero-mid) <= abs(right_zero-mid):
                        cut_index = left_zero
                    else:
                        cut_index = right_zero
                elif left_zero != -1:
                    cut_index = left_zero
                elif right_zero != -1:
                    cut_index = right_zero

                # Check for VP = MFV second
                elif VP[mid] == MFV:
                    cut_index = mid
                elif left_MFV != -1:
                    cut_index = left_MFV
                elif right_MFV != -1:
                    cut_index = right_MFV
                else:
                    cut_index = mid


                separation_regions.append((end, cut_index, start))

        i -= 1

    return separation_regions


def check_baseline(word_img, start, end, upper_base, lower_base):
    
    j = end+1

    cnt = 0
    while j < start:
    
        # Black pixel (Discontinuity)
        base = upper_base
        while base <= lower_base:
            
            pixel = word_img[base][j]
            cnt += pixel

            base += 1
        
        j += 1

    if cnt == 0:
        return False

    return True


def check_hole(segment):
    '''Check if a segment has a hole or not'''

    contours, hierarchy = cv.findContours(segment, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cnt = 0
    for hier in hierarchy[0]:
        if hier[3] >= 0:
            cnt += 1
    
    if cnt == 0:
        return True

    return False


def check_stroke(segment, upper_base, lower_base):
    '''Check if the segment is a stroke or not'''

    nodots_segment = segment.copy()
    erase_points(nodots_segment, upper_base)    
    
    components, labels= cv.connectedComponents(nodots_segment, connectivity=8)
    
    # Background and another component
    if components == 2:

        segment_HP = projection(nodots_segment, 'horizontal')

        # Above sum from start to upper baseline
        SHPA = np.sum(segment_HP[:upper_base])
        # Below sum from lower basleine to end
        SHPB = np.sum(segment_HP[lower_base+1:])

        # TODO
        if SHPA > SHPB and not(check_hole(nodots_segment)):
            pass 


def filter_regions(word_img, SRL:list, VP:list, upper_base:int, lower_base:int, MTI:int, MFV:int, top_line:int):
    
    valid_separation_regions = []
    T = 1

    SR_idx = 0
    while SR_idx < len(SRL):
        
        SR = SRL[SR_idx]
        end_idx, cut_idx, start_idx = SR

        # Case 1 : Vertical Projection = 0
        if VP[cut_idx] == 0:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue

        components, labels= cv.connectedComponents(word_img, connectivity=8)

        # Case 2 : no connected path between start and end
        if labels[MTI, end_idx] != labels[MTI, start_idx]:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue

        # Case 3 : Contain Holes
        j = end_idx + 1
        f = 1
        while j < start_idx:
            VT = vertical_transitions(word_img, j)
            if VT <= 2:
                f = 0
                break
            j += 1
            
        if f == 1:
            SR_idx += 1
            continue

        # ***Case 4 : No baseline between start and end
        j = end_idx+1

        cnt = 0
        while j < start_idx:
            
            # Black pixel (Discontinuity)
            base = upper_base
            while base <= lower_base:
                
                pixel = word_img[base][j]
                cnt += pixel

                base += 1
            
            j += 1

        segment = word_img[:, end_idx: start_idx+1]

        if cnt == 0:

            segment_HP = projection(segment, 'horizontal')

            SHPA = np.sum(segment_HP[:upper_base])
            SHPB = np.sum(segment_HP[lower_base+T+1:])

            if SHPB > SHPA:
                SR_idx += 1
                continue
            elif VP[cut_idx] <= MFV + T:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
            else:
                SR_idx += 1
                continue
        
        # Case 5 : Last region or next VP[cut] = 0
        if SR_idx == len(SRL) - 1 or VP[SRL[SR_idx+1][1]] == 0:

            if SR_idx == len(SRL) - 1:
                segment = word_img[:, :end_idx+1]
            else:
                next_cut = SRL[SR_idx+1][1]
                segment = word_img[:, next_cut:end_idx+1]

            segment_HP = projection(segment, 'horizontal')
            (h, w) = segment.shape
            # SHPA = np.sum(segment_HP[:upper_base])
            # SHPB = np.sum(segment_HP[lower_base+1:])

            # if SHPA > SHPB:
            #     SR_idx += 1
            #     continue

            top_left_pixel = -1
            for col in range(w):
                for row in range(h):
                    if segment[row][col] == 1:
                        top_left_pixel = row
                        break
                if top_left_pixel != -1:
                    break

            # for i, proj in enumerate(segment_HP):
            #     if proj != 0:
            #         top_left_pixel = i
            #         break

            # if SR_idx == 0:
            #     breakpoint()

            if SR_idx == len(SRL) - 1:
                if upper_base - top_left_pixel <= (upper_base-top_line)/2 and upper_base - top_left_pixel >= 0:
                    SR_idx += 1
                    continue
            else:
                if 0 <= upper_base - top_left_pixel <= 2:
                    SR_idx += 1
                    continue
                
        # Strokes 

        next_cut = SRL[SR_idx+1][1]
        segment = word_img[:, next_cut:cut_idx+1]

        # Case 6 : SEG is no Stroke
        if not(check_stroke(segment, upper_base, lower_base)):

            next_end = SRL[SR_idx+1][0]
            next_start = SRL[SR_idx+1][2]

            if not(check_baseline(word_img, next_start, next_end, upper_base, lower_base) and VP[next_cut] <= MFV+T):
                SR_idx += 1
                continue
            else:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
        
        components, labels = cv.connectedComponents(segment, connectivity=8)
        
        # Case 7 : stroke with dots above or below
        if components > 2:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue


        valid_separation_regions.append(SR)
        SR_idx += 1

    return valid_separation_regions


def segment(line, word_img):

    binary_word = binarize(word_img)
    l = binary_word.copy()

    VP = projection(binary_word, 'vertical')
    # MFV = np.argmax(np.bincount(VP))
    upper_base, lower_base, MFV = baseline_detection(binary_word)
    MTI = horizontal_transitions(binary_word, upper_base)

    SRL = cut_points(binary_word, VP, MFV, MTI)
    HP = projection(line, 'horizontal')
    top_line = -1
    for i, proj in enumerate(HP):
        if proj != 0:
            top_line = i
            break

    valid = filter_regions(binary_word, SRL, VP, upper_base, lower_base, MTI, MFV, top_line)

    V = np.dstack([l*255, l*255, l*255])
    for region in valid:
        V[:, region[1], :] = [255, 0, 0]

    print(MFV)
    print(SRL)
    print(valid)

    plt.imshow(V)
    plt.show()


if __name__ == "__main__":
    
    img = cv.imread('../Dataset/scanned/capr1.png')
    lines = line_horizontal_projection(img)

    line = lines[2]
    words = word_vertical_projection([line])[0]

    word = words[1]

    # breakpoint()
    segment(line, word)
