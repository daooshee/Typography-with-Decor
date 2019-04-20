import os



''' List of style and content images
    * For each style image, for example: XmasM, there should be a Input/style/XmasM.png (the input style image)
    and a Input/style/XmasM_glyph.png (the corresponding glyph of Input/style/XmasM.png, pre-processed)
    * For each content image, for example: x, there shoud be a Input/content/x.png (pre-processed)
'''
Style = ['Face', 'Snowman', 'Xmas','XmasM', 'XmasM', 'XmasM', 'XmasM']
Content = ['lan','ha','virgo','x','m','a','s']



''' Some other kind of parameters
    gpu_id: gpu id, -1 for cpu. No support for multi-gpu. If you want multi-gpu, you have to rewrite some .py scripts.
    batch_size: batch size for one-shot fine-tuning.
    finetuning_iteration: one-shot fine-tuning iteration. Should > 100.
'''

gpu_id = 0
batch_size = 48
finetuning_iteration = 200



''' Processing
'''
for img_id, style_name in enumerate(Style):
    content_name = Content[img_id]

    print("-----------------------------------------")
    print("       Style: %s"%(style_name))
    print("       Content: %s"%(content_name))
    print("-----------------------------------------")

    # Detect decors 
    print("\n--- Decorative Element Segmentation ---")
    # Some temporal images are stored in temp/
    if os.path.exists('temp/'):
        os.system('rm -r temp/')
    os.mkdir('temp')

    os.system('python Segmentation/segmentation.py --img Input/style/%s.png --img_content Input/style/%s_glyph.png'%(style_name,style_name))
    os.system('python Segmentation/crf_post_process.py --img Input/style/%s.png'%(style_name))
    print('Successfully generate mask_final.jpg and mask_ori.jpg.')

    if not (os.path.exists('temp/mask_final.jpg') and os.path.exists('temp/mask_ori.jpg')):
        exit('Error, cannot find mask_final.jpg and mask_ori.jpg.')

    # One-shot fine-tuning
    print("\n--- One-Shot Fine-Tuning ---")
    if os.path.exists('cache/%s_netG.pth'%(style_name)):
        print('Exists %s_netG.pth. Skip one-shot fine-tuning.'%(style_name))
    else:
        os.system('python FineTuning/train.py --gpu %d --batchSize %d --niter %d\
                        --netf pre-trained \
                        --style_name %s \
                        --style_path Input/style/%s.png \
                        --glyph_path Input/style/%s_glyph.png \
                        ' % (gpu_id, batch_size, finetuning_iteration, style_name, style_name, style_name))

    # Basal text effects transfer
    print("\n--- Basal Text Effect Transfer---")
    if os.path.exists('basal-text-effect-transfer/%s.png'%(style_name+'_'+content_name)):
        print('Exists %s.png. Skip basal text effect transfer.'%(style_name+'_'+content_name))
    else:
        os.system('python FineTuning/generate.py --gpu %d\
                        --outf basal-text-effect-transfer/ \
                        --style_name %s \
                        --style_path Input/style/%s.png \
                        --glyph_path Input/style/%s_glyph.png \
                        --content_path Input/content/%s.png \
                        --save_name %s' % (gpu_id, style_name, style_name, style_name, content_name, style_name+'_'+content_name))

    if not os.path.exists('basal-text-effect-transfer/%s.png'%(style_name+'_'+content_name)):
        exit('Error, cannot find text effect transfer result.')

    # Recombine decors and basal text effects
    print("\n--- Decor Recomposition ---")
    os.system('python decor_recomposition.py \
                    --style_path Input/style/%s.png \
                    --glyph_path Input/style/%s_glyph.png \
                    --content_path Input/content/%s.png \
                    --transfered_path basal-text-effect-transfer/%s.png \
                    --save_path result/%s.png \
                    --style_name %s \
                    --content_name %s' % (style_name, style_name, content_name, style_name+'_'+content_name, \
                                        style_name+'_'+content_name, style_name, content_name))
