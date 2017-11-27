import os

Src = os.path.dirname(os.path.abspath(__file__)) # src directory
Root = os.path.dirname(Src) + '/' # root directory
Src = Src + '/'
Models = os.path.join(Root, 'models') + '/'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model_statistics(model, stats_file):
    # save model size, flops, activations by layers
    with open(stats_file, 'w+') as f:
        for counter, label in (model.model_size_counter, '# params'), (model.activation_counter, 'Activation size'), (model.flop_counter, '# flops'):
            count = 0
            f.write('%s by layer:\n' % label)
            for c in counter:
                f.write('\t%s: %s\n' % (c[0], c[1]))
                count += c[1]
            f.write('\ttotal: %s\n\n' % count)
    print('Model statistics saved to %s' % stats_file)
    
def draw_box(im, box_list, label_list, color=(0,255,0), cdict={}, form='center'):
    assert form in ['center', 'diagonal'], 'bounding box format not accepted: %s.' % form

    for bbox, label in zip(box_list, label_list):
        if form == 'center':
            bbox = bbox_transform(bbox)

        xmin, ymin, xmax, ymax = [int(b) for b in bbox]

        l = label.split(':')[0] # text before "CLASS: PROB"
        c = cdict.get(l, color)

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)