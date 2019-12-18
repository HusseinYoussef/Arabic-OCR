def word_vertical_projection(line_images:list, cut=3):
    
    lines_words = []
    for line_image in line_images:
        VP = projection(line_image, 'vertical')
        Avg = 0
        Len = 0
        Cnt = 0
        for i in range(len(VP)):
            while(i<len(VP) and VP[i]==0):
                Len+=1
                i+=1
            if(Len<=10 and Len>0):
                Cnt += (Len>0)
                Avg += Len
                #print(Len)
            Len = 0
            i-=1
        Avg = Avg//Cnt
        print(Avg)
        line_words = projection_segmentation(line_image, axis='vertical', cut=Avg)
        lines_words.append(line_words)

    return lines_words

    def clear_file(filepath,extension = ".png"):
    filelist = [ f for f in os.listdir(filepath) if f.endswith(extension) ]
    for f in filelist:
        os.remove(os.path.join(filepath, f))