import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utilities import projection
from preprocessing import erase_points
from segmentation import line_horizontal_projection, word_vertical_projection
from skimage.morphology import skeletonize, thin


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
    line_idx = baseline_idx-1
    lines = []
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
                
        if current_transitions > max_transitions:
            max_transitions = current_transitions
            lines.append(line_idx)
            max_transitions_idx = line_idx

        line_idx -= 1
    
    # if len(lines) > 1:
    #     return lines[1]
    # else:
    #     return lines[0]
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
                elif VP[mid] <= MFV+T:
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


def inside_hole(word_img, end_idx, start_idx):
    '''Check if a segment has a hole or not'''

    if end_idx == 0 and start_idx == 0:
        return 0
    j = end_idx + 1
    flag = 1
    while j < start_idx:
        VT = vertical_transitions(word_img, j)
        if VT <= 2:
            flag = 0
            break
        j += 1
    
    return flag


def check_hole(segment):
    '''Check if a segment has a hole or not'''

    # no_dots = segment.copy()
    # erase_points(no_dots, upper_base)

    contours, hierarchy = cv.findContours(segment, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnt = 0
    for hier in hierarchy[0]:
        if hier[3] >= 0:
            cnt += 1

    return cnt != 0


def remove_dots(word_img, threshold=12):

    no_dots = word_img.copy()

    components, labels, stats, GoCs = cv.connectedComponentsWithStats(no_dots, connectivity=8)
    char = []
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if size > threshold:
            char.append(label)
    for label in range(1, components):
        _, _, _, _, size = stats[label]
        if label not in  char:
            no_dots[labels == label] = 0

    return no_dots


def check_dots(segment):

    contours, hierarchy = cv.findContours(segment, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = 0
    for c in contours:
        if len(c) >= 2:
            cnt +=1 
    return cnt > 1


def check_stroke(no_dots_copy, segment, upper_base, lower_base, SR1, SR2):

    T = 1
    components, labels, stats, cen= cv.connectedComponentsWithStats(segment, connectivity=8)
    skeleton = skeletonize(segment.copy())
    (h, w) = segment.shape

    cnt = 0
    for c in range(1, components):
        if stats[c][4] > 3:
            cnt += 1

    if cnt > 2:
        return False

    if check_hole(segment) or inside_hole(no_dots_copy, SR1[0], SR1[1]) or inside_hole(no_dots_copy, SR2[0], SR2[1]):
        return False

    HP = projection(skeleton, 'horizontal')
    VP = projection(segment, 'vertical')

    seg_l = -1
    seg_r = -1
    for i in range(0, len(VP)):
        if VP[i] != 0:
            seg_l = i
            break
    for i in range(len(VP)-1, -1, -1):
        if VP[i] != 0:
            seg_r = i
            break

    seg_width = seg_r - seg_l + 1
    SHPA = np.sum(HP[:upper_base])
    SHPB = np.sum(HP[lower_base+T+1:])
    MFV_HP = np.argmax(np.bincount(HP)[1:])+1
    MFV = lower_base - upper_base + 1 + T

    top_pixel = -1
    for i, proj in enumerate(HP[:upper_base]):
        if proj != 0:
            top_pixel = i
            break
    height = upper_base-top_pixel
    
    VT = 0
    for i in range(w):
        if vertical_transitions(segment, i) > 2:
            VT += 1

    if SHPB == 0 and height <= 6 and VT <= 1 and abs(MFV - MFV_HP) <= 2 and seg_width <= 6:
        return True

    return False


def check2_stroke(no_dots_copy, segment, upper_base, lower_base, SR1, SR2):
    '''Check if the segment is a stroke or not'''
    
    T = 1
    components, labels, stats, cen= cv.connectedComponentsWithStats(segment, connectivity=8)
    cnt = 0
    for c in range(components):
        if stats[c][4] > 5:
            cnt += 1
    
    if cnt > 2:
        return False

    # if check_hole(segment):
    if check_hole(segment) or inside_hole(no_dots_copy, SR1[0], SR1[1]) or inside_hole(no_dots_copy, SR2[0], SR2[1]):
        return False

    segment_HP = projection(segment, 'horizontal')
    height = -1
    for i, proj in enumerate(segment_HP[:upper_base]):
        if proj != 0:
            height = i
            break

    MFV =  np.argmax(np.bincount(segment_HP[1:]))+1
    base_thick = abs(upper_base-lower_base)+1 + T

    SHPA = np.sum(segment_HP[:upper_base])
    SHPB = np.sum(segment_HP[lower_base+T+1:])

    (h, w) = segment.shape

    VP = projection(segment, 'vertical')
    skeleton = skeletonize(segment)[:upper_base, :].astype(np.uint8)
    components, labels, stats, cen= cv.connectedComponentsWithStats(skeleton, connectivity=8)
    
    if components > 2:
        for label in range(components):
            if stats[label][4] == 1:
                skeleton[labels == label] = 0

    VP_above = projection(skeleton[:upper_base, :], 'vertical')
    HP = projection(segment, 'horizontal')
    MFV_HP = np.argmax(np.bincount(HP)[1:])+1
    peak = np.amax(HP)
    second_peak = 0
    stroke_lft = -1
    stroke_right = -1

    j = 0
    while j < w:
        if VP_above[j] > 0:
            stroke_lft = j
            break
        j += 1

    j = w-1
    while j >= 0:
        if VP_above[j] > 0:
            stroke_right = j
            break
        j -= 1

    for val in HP:
        if val < peak and val > second_peak:
            second_peak = val
    j = 0
    while j < w:
        if VP[j] != 0:
            lft = j
            break
        j += 1

    j = w-1
    while j >= 0:
        if VP[j] != 0:
            right = j
            break
        j -= 1

    # breakpoint ()
    if (SHPB == 0 or (int(SHPA) - int(SHPB)) >= 4) and abs(base_thick-MFV <=2) \
        and right-lft+1 <= 6 and 1 <= stroke_right-stroke_lft+1 <= 3:
            if 1 <= stroke_right-stroke_lft+1 < 3:
                if (upper_base-height) <= 6 :
                    return True
            elif stroke_right-stroke_lft+1 == 3 and (upper_base-height) < 6: 
                return True

    return False


def filter_regions(word_img, SRL:list, VP:list, upper_base:int, lower_base:int, MTI:int, MFV:int, top_line:int):
    
    valid_separation_regions = []
    overlap = []
    no_dots_copy = remove_dots(word_img)

    T = 1
    components, labels= cv.connectedComponents(word_img[:lower_base+5, :], connectivity=8)

    SR_idx = 0
    while SR_idx < len(SRL):
        
        SR = SRL[SR_idx]
        end_idx, cut_idx, start_idx = SR

        # Case 1 : Vertical Projection = 0
        if VP[cut_idx] == 0:
            valid_separation_regions.append(SR)
            SR_idx += 1
            continue


        # Case 2 : no connected path between start and end
        # components, labels= cv.connectedComponents(word_img[:, end_idx:start_idx+1], connectivity=8)
        if labels[MTI, end_idx] != labels[MTI, start_idx]:
            valid_separation_regions.append(SR)
            overlap.append(SR)
            SR_idx += 1
            continue

        # Case 3 : Contain Holes
        # if SR_idx == 3:
        #     breakpoint()

        if inside_hole(no_dots_copy, end_idx, start_idx):
            SR_idx += 1
            continue

        # ***Case 4 : No baseline between start and end
        segment = word_img[:, end_idx: start_idx+1]

        j = end_idx+1
        cnt = 0
        while j < start_idx:
            
            # Black pixel (Discontinuity)
            base = upper_base
            while base <= lower_base+T:
                
                pixel = word_img[base][j]
                cnt += pixel

                base += 1
            
            j += 1


        if cnt == 0:
            
            segment_HP = projection(segment, 'horizontal')

            SHPA = np.sum(segment_HP[:upper_base])
            SHPB = np.sum(segment_HP[lower_base+T+1:])

            if (int(SHPB) - int(SHPA)) >= 0:
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
                segment = no_dots_copy[:, :end_idx+1]
                next_cut = 0
            else:
                next_cut = SRL[SR_idx+1][1]
                segment = no_dots_copy[:, next_cut:end_idx+1]

            segment_HP = projection(segment, 'horizontal')
            (h, w) = segment.shape

            top_left_pixel = -1
            dist = -1
            for col in range(w):
                for row in range(h):
                    if segment[row][col] == 1:
                        top_left_pixel = row
                        dist = col
                        break
                if top_left_pixel != -1:
                    break

            # if SR_idx == len(SRL) - 1:
                # breakpoint()
                    
            SHPA = np.sum(segment_HP[:upper_base])
            SHPB = np.sum(segment_HP[lower_base+T+1:])   
            if ((1 <= upper_base - top_left_pixel <= (upper_base-top_line)/2 and upper_base - top_left_pixel >= 0)\
                or (int(SHPB) - int(SHPA)) > 4 \
                or ((0 <= upper_base - top_left_pixel <= 2) and ((cut_idx - (dist+next_cut)) <= 5)))\
                and not check_hole(segment):
                SR_idx += 1
                continue
            # else:
            #     valid_separation_regions.append(SR)
            #     SR_idx += 1
            #     continue
            # else:
            #     if 0 <= upper_base - top_left_pixel <= 2 and cut_idx - (dist+next_cut) <= 5:
            #         SR_idx += 1
            #         continue
                
        # Strokes 

        SEGP = (-1, -1)
        SEG = (-1, -1)
        SEGN = (-1, -1)
        SEGNN = (-1, -1)
        SEGP_SR1 = (0, 0)
        SEGP_SR2 = (0, 0)
        SEG_SR1 = (0, 0)
        SEG_SR2 = (0, 0)
        SEGN_SR1 = (0, 0)
        SEGN_SR2 = (0, 0)
        SEGNN_SR1 = (0, 0)
        SEGNN_SR2 = (0, 0)

        current_cut = SR[1]
     
        if SR_idx == 0:
            SEGP = (SRL[SR_idx][1], word_img.shape[1]-1)
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx][1], word_img.shape[1]-1)

        if SR_idx > 0:
            SEGP = (SRL[SR_idx][1], SRL[SR_idx-1][1])
            SEGP_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEGP_SR2 = (SRL[SR_idx-1][0], SRL[SR_idx-1][2])
        
        if SR_idx < len(SRL)-1:
            SEG = (SRL[SR_idx+1][1], SRL[SR_idx][1])
            SEG_SR1 = (SRL[SR_idx][0], SRL[SR_idx][2])
            SEG_SR2 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])

        if SR_idx < len(SRL)-2:
            SEGN = (SRL[SR_idx+2][1], SRL[SR_idx+1][1])
            SEGN_SR1 = (SRL[SR_idx+1][0], SRL[SR_idx+1][2])
            SEGN_SR2 = (SRL[SR_idx+2][0], SRL[SR_idx+2][2])
        
        if SR_idx < len(SRL)-3:
            SEGNN = (SRL[SR_idx+3][1], SRL[SR_idx+2][1])
            SEGNN_SR1 = (SRL[SR_idx+2][0], SRL[SR_idx+2][2])
            SEGNN_SR2 = (SRL[SR_idx+3][0], SRL[SR_idx+3][2])

        
        # if SR_idx == 5:
        #     breakpoint()
        
        # SEG is stroke with dots
        if SEG[0] != -1 and\
            (check_stroke(no_dots_copy, no_dots_copy[:, SEG[0]:SEG[1]], upper_base, lower_base, SEG_SR1, SEG_SR2) \
            and check_dots(word_img[:, SEG[0]:SEG[1]])):
            
            # breakpoint()
            # Case when starts with ش
            if SEGP[0] != -1 and \
                ((check_stroke(no_dots_copy, no_dots_copy[:, SEGP[0]:SEGP[1]], upper_base, lower_base, SEGP_SR1, SEGP_SR2) \
                and not check_dots(word_img[:, SEGP[0]:SEGP[1]]))\
                and (SR_idx == 0 or VP[SRL[SR_idx-1][1]] == 0 or (VP[SRL[SR_idx-1][1]] == 0 and SRL[SR_idx-1] in overlap))):
                
                SR_idx += 2
                continue
            else:
                valid_separation_regions.append(SR)
                SR_idx += 1
                continue
                
        # SEG is stroke without dots
        elif SEG[0] != -1 and (check_stroke(no_dots_copy, no_dots_copy[:, SEG[0]:SEG[1]], upper_base, lower_base, SEG_SR1, SEG_SR2) \
            and not check_dots(word_img[:, SEG[0]:SEG[1]])):

            # Case starts with س
            if SEGP[0] != -1 and (check_stroke(no_dots_copy, no_dots_copy[:, SEGP[0]:SEGP[1]], upper_base, lower_base, SEGP_SR1, SEGP_SR2) \
                and not check_dots(word_img[:, SEGP[0]:SEGP[1]])):

                SR_idx += 2
                continue

            # SEGN is stroke without dots
            if SEGN[0] != -1 \
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2) \
                and not check_dots(word_img[:, SEGN[0]:SEGN[1]])):

                valid_separation_regions.append(SR)
                SR_idx += 3
                continue

            # SEGN stroke with Dots and SEGNN stroke without Dots
            if SEGN[0] != -1\
                and (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2) \
                and check_dots(word_img[:, SEGN[0]:SEGN[1]]) \
                and SEGNN[0] != -1 \
                and check_stroke(no_dots_copy, no_dots_copy[:, SEGNN[0]:SEGNN[1]], upper_base, lower_base, SEGNN_SR1, SEGNN_SR2) \
                and not check_dots(word_img[:, SEGNN[0]:SEGNN[1]])):
        
                    valid_separation_regions.append(SR)
                    SR_idx += 3
                    continue
            
            # SEGN is not stroke or Stroke with Dots
            if SEGN[0] != -1 \
                and ((not check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2)) \
                or (check_stroke(no_dots_copy, no_dots_copy[:, SEGN[0]:SEGN[1]], upper_base, lower_base, SEGN_SR1, SEGN_SR2) \
                and check_dots(word_img[:, SEGN[0]:SEGN[1]]))):
                    
                    SR_idx += 1
                    continue


        if (len(valid_separation_regions) == 0 or\
            len(valid_separation_regions) > 0 and abs(cut_idx-valid_separation_regions[-1][1]) > 2): 
            valid_separation_regions.append(SR)
        SR_idx += 1

    return valid_separation_regions


def segment(line, word_img):

    binary_word = binarize(word_img)
    # binary_word = word_img
    l = binary_word.copy()

    sk = skeletonize(l)

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
    print(upper_base)
    print(lower_base)
    print(SRL)
    print(valid)

    plt.imshow(V, 'gray')
    plt.show()


if __name__ == "__main__":
    
    img = cv.imread('../Dataset/scanned/capr1.png')
    lines = line_horizontal_projection(img)

    line = lines[7]
    words = word_vertical_projection([line])[0]

    word = words[9]

    # breakpoint()
    segment(line, word)
