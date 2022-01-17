'''to test smplx model posed and shaped.'''

# 
from email.headerregistry import ContentTypeHeader
from logging import root
from readline import insert_text
import bpy
import numpy as np 
from mathutils import Matrix, Quaternion,Vector
import json
import math
import mathutils
import os
import sys
#
# two class hand pose 

minz=0

hand_pose={}
hand_pose['flat']= np.zeros([90]).reshape(2,-1)
hand_pose['relaxed'] = np.array(
                        [[ 0.11167871,  0.04289218, -0.41644183,  0.10881133, -0.06598568,
                        -0.75622   , -0.09639297, -0.09091566, -0.18845929, -0.11809504,
                        0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.7048571 ,
                        -0.01918292, -0.09233685, -0.3379135 , -0.45703298, -0.19628395,
                        -0.6254575 , -0.21465237, -0.06599829, -0.50689423, -0.36972436,
                        -0.06034463, -0.07949023, -0.1418697 , -0.08585263, -0.63552827,
                        -0.3033416 , -0.05788098, -0.6313892 , -0.17612089, -0.13209307,
                        -0.37335458,  0.8509643 ,  0.27692273, -0.09154807, -0.49983943,
                        0.02655647,  0.05288088,  0.5355592 ,  0.04596104, -0.27735803],
                        [0.11167871, -0.04289218,  0.41644183,  0.10881133,  0.06598568,
                        0.75622   , -0.09639297,  0.09091566,  0.18845929, -0.11809504,
                        -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.7048571 ,
                        -0.01918292,  0.09233685,  0.3379135 , -0.45703298,  0.19628395,
                        0.6254575 , -0.21465237,  0.06599829,  0.50689423, -0.36972436,
                        0.06034463,  0.07949023, -0.1418697 ,  0.08585263,  0.63552827,
                        -0.3033416 ,  0.05788098,  0.6313892 , -0.17612089,  0.13209307,
                        0.37335458,  0.8509643 , -0.27692273,  0.09154807, -0.49983943,
                        -0.02655647, -0.05288088,  0.5355592 , -0.04596104,  0.27735803]])


part_match={ 
            'root': 'root', 'bone_00':  'pelvis', 'bone_01':  'left_hip', 'bone_02':  'right_hip', 
            'bone_03':  'spine1', 'bone_04':  'left_knee', 'bone_05':  'right_knee', 'bone_06':  'spine2', 
            'bone_07':  'left_ankle', 'bone_08':  'right_ankle', 'bone_09':  'spine3', 'bone_10':  'left_foot', 
            'bone_11':  'right_foot', 'bone_12':  'neck', 'bone_13':  'left_collar', 'bone_14':  'right_collar', 
            'bone_15':  'head', 'bone_16':  'left_shoulder', 'bone_17':  'right_shoulder', 'bone_18':  'left_elbow', 
            'bone_19':  'right_elbow', 'bone_20':  'left_wrist', 'bone_21':  'right_wrist', 'bone_22':  'jaw', 
            'bone_23':  'left_eye_smplhf', 'bone_24':  'right_eye_smplhf', 
            'bone_25':  'left_index1', 'bone_26':  'left_index2', 'bone_27':  'left_index3', 
            'bone_28':  'left_middle1', 'bone_29':  'left_middle2', 'bone_30':  'left_middle3', 
            'bone_31':  'left_pinky1', 'bone_32':  'left_pinky2', 'bone_33':  'left_pinky3', 
            'bone_34':  'left_ring1', 'bone_35':  'left_ring2', 'bone_36':  'left_ring3', 
            'bone_37':  'left_thumb1', 'bone_38':  'left_thumb2', 'bone_39':  'left_thumb3', 
            'bone_40':  'right_index1', 'bone_41':  'right_index2', 'bone_42':  'right_index3', 
            'bone_43':  'right_middle1', 'bone_44':  'right_middle2', 'bone_45':  'right_middle3', 
            'bone_46':  'right_pinky1', 'bone_47':  'right_pinky2', 'bone_48':  'right_pinky3', 
            'bone_49':  'right_ring1', 'bone_50':  'right_ring2', 'bone_51':  'right_ring3', 
            'bone_52':  'right_thumb1', 'bone_53':  'right_thumb2', 'bone_54':  'right_thumb3'
            }

def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0.0 else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat



def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(-1, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate(
        [(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]]
    )
    return mat_rots, bshapes


def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return


def apply_only_shape_get_minz(shape,ob):
    trans=np.zeros([0,0,0]) # root location at [0,0,0]
    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
   
    # 



def apply_trans_pose_shape(orient,trans,pose,shape,arm_ob,ob,delta_orginal,minz,frame=None):
    """
    Apply trans pose and shape to character
    """

    # set global orientation
    rot=np.array([-1.5707963267948966,0,0])
    set_pose_from_rodrigues(arm_ob,'root',rot)  
    

    # t=np.zeros(3)S
    # t[0]=-trans[2]
    # t[1]=-trans[0]
    # t[2]=trans[1]

    # transform pose into rotation matrices (for pose) and pose blendshapes            
    mrots, bsh = rodrigues2bshapes(pose)

    # rot=np.array([-1.5707963267948966,0,0])
    # root_pose=Rodrigues(pose[:3])
    rot_mat=Rodrigues(rot)
    # rot_mat=np.dot(root_pose,rot_mat)
    delta_ori = np.dot(rot_mat,delta_orginal)
    trans=np.dot(rot_mat,trans)+delta_ori
    
    # set the location of the first bone to the translation parameter
    # arm_ob.pose.bones['pelvis'].location = trans
    # arm_ob.pose.bones['root'].location = trans
    if frame is not None:
        # arm_ob.pose.bones['pelvis'].location = trans
        arm_ob.pose.bones['root'].location = trans
        arm_ob.pose.bones['root'].keyframe_insert('location', frame=frame)

    # set the pose of each bone to the quaternion specified by pose
    for ibone, mrot in enumerate(mrots):
        bone = arm_ob.pose.bones[part_match['bone_%02d' % ibone]]
        bone.rotation_quaternion = Matrix(mrot).to_quaternion()
        if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            bone.keyframe_insert('location', frame=frame)

    # apply pose blendshapes
    for ibshape, bshape in enumerate(bsh):
        ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
        if frame is not None:
            ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert(
                'value', index=-1, frame=frame)
    # apply shape blendshapes
    for ibshape, shape_elem in enumerate(shape):
        ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
        if frame is not None:
            ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert(
                'value', index=-1, frame=frame)



    # vertices_world = [vertex.co for vertex in ob.data.vertices]
    # z_min = (min(vertices_world, key=lambda item: item.z)).z
    # print(z_min)
    get_minz()

    
    

   
    # set_pose_from_rodrigues(arm_ob,'root',rot)
    # apply face shape blendshapes
    # if isinstance(expression,type(None)):
    #     expression=[0.0]*10
    # for ibshape, shape_elem in enumerate(expression):
    #     ob.data.shape_keys.key_blocks['Exp%03d' % ibshape].value = shape_elem
    #     if frame is not None:
    #         ob.data.shape_keys.key_blocks['Exp%03d' % ibshape].keyframe_insert(
    #             'value', index=-1, frame=frame)

   
def radians2angle(radians):
    #radians -> angle
    pi = 180.0 / math.pi
    return radians*pi

def angle2radians(angle):
    # angle -> radians
    pi = math.pi / 180.0
    return angle*pi 


def setupCamera(camera, c):
    pi = math.pi
    camera.rotation_mode = 'XYZ'

    camera.rotation_euler[0] = 0
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 0

    camera.rotation_euler[0] = c[0]
    camera.rotation_euler[1] = c[1]
    camera.rotation_euler[2] = c[2]

    camera.location.x = c[3]
    camera.location.y = c[4]
    camera.location.z = c[5]

    return

def create_cameras_array(H,R,number_rank,number_per_rank,ground_z):
    '''
    @input:
        H: the cameras array's height
        R: the cameras array's max circle radius
        number_rank: the number of different height cameras' circles
        number_per_rank: camera number of per cameras' circle 
        ground_z: z-axis location for ground, the height of camera array is 0.
    @return:
        camera_queue: the cameras array queue.
    '''
    context = bpy.context
    scene = context.scene
    coll = bpy.data.collections.new("CameraArray") # 
    scene.collection.children.link(coll) 

    camera_queue = []
    #
    def create_cam(h,r,num_camera,project_h,ith_rank):
        # create a camera circle. 
        # set camera and copy 
        angle_x = math.atan2(project_h-h,r) + math.radians(90.0)
        rot_loc=[angle_x,0,math.radians(90.0),r,0,h]
        # setupCamera(scene.camera,rot_loc)
        rot_mat = mathutils.Euler((rot_loc[0], rot_loc[1], rot_loc[2]), 'XYZ')
        rot_mat_np=np.array(rot_mat.to_matrix())
        world_matrix=np.eye(4)
        world_matrix[:-1,:-1] = rot_mat_np
        world_matrix[:-1,-1] = rot_loc[3:]

        # transfrom to Matrix
        world_matrix_math = Matrix(world_matrix)

        angle_interval = 360.0 // num_camera

        begin_angle = 0
        if ith_rank%2==0:
            begin_angle +=angle_interval/2
        for i in range(num_camera):
            cam_copy= scene.camera.copy()
            R = Matrix.Rotation(math.radians(angle_interval*i+begin_angle), 4, 'Z')
            cam_copy.matrix_world = R @ world_matrix_math
            coll.objects.link(cam_copy)
            camera_queue.append(cam_copy)

    # mean  height
    height_interval = (2*H) / number_rank 

    
    for i in range(number_rank):
        h= i*height_interval
        r=R
        if h>H:
            # r=pow(R**2-(h-H)**2,0.5)
            r = R*(4-h)/2
        h +=ground_z
        create_cam(h,r,number_per_rank,0,i)

    return camera_queue
        
    


def load_pose_info(file_path):
    with open(file_path) as fp:
        data=json.load(fp)

    Rhs=np.array(data['Rh']).reshape(-1,3)
    Ths=np.array(data['Th']).reshape(-1,3)
    poses=np.array(data['poses']).reshape(-1,87)
    expressions=np.array(data['expressions'])
    shapes=np.array(data['shapes'])

    # pose [22*3,12,3*3]
    num_body_joints=22*3
    num_hand_joints=6*2
    num_face_joints=3*3

    num_frame=poses.shape[0]

    trans_pose=[]
    trans_shapes=[]
    trans_trans=[]
    trans_orient=[]
    trans_expression=[]

    for i in range(num_frame):
        pose=np.zeros([1,55*3])
        pose[:,:num_body_joints]=poses[i][:num_body_joints] # body
        pose[:,num_body_joints:num_body_joints+num_face_joints] = poses[i][num_body_joints+num_hand_joints:] # face
        pose[:,num_body_joints+num_face_joints:]=hand_pose['relaxed'].reshape(-1) # left hand and right hand
        t=Ths[i].reshape(3)
        r=Rhs[i].reshape(3)
        # r=np.zeros(3).reshape(1,3)
        shape=shapes[i].reshape(10)
        expression=expressions[i].reshape(10)
        pose=pose.reshape(-1)
        #
        trans_pose.append(pose)
        trans_shapes.append(shape)
        trans_trans.append(t)
        trans_orient.append(r)
        trans_expression.append(expression)

    return trans_pose,trans_shapes,trans_trans,trans_orient,trans_expression


def load_pose_from_ACCAD(file_path):
    with np.load(file_path) as data:
        poses=data['poses']
        trans=data['trans']
        orient=data['root_orient']
        shapes=data['betas']

    return poses,trans,orient,shapes


def init_scene(model_dir,ground=0):
    bpy.ops.import_scene.fbx(filepath=model_dir, axis_forward="Y",axis_up="Z",global_scale=1)
    for mesh in bpy.data.objects.keys():
        if mesh == 'SMPLX-mesh-male':
            ob = bpy.data.objects[mesh]
            arm_obj = 'SMPLX-male'   
        if mesh == 'SMPLX-mesh-female':
            ob = bpy.data.objects[mesh]
            arm_obj = 'SMPLX-female'
        if mesh == 'SMPLX-mesh-neutral':
            ob = bpy.data.objects[mesh]
            arm_obj = 'SMPLX-neutral' 

    arm_ob = bpy.data.objects[arm_obj] 

    # create camera array
    _=create_cameras_array(2,8,3,4,ground)

    # new light
    new_light(8,4)
    # hdri
    Env_hdri('/Volumes/Samsung_T5/data/test_blender/HDRI/abandoned_factory_canteen_01_4k.exr')

    # material 
    texture_mesh(0,'/Volumes/Samsung_T5/data/test_blender/smplx_uv/nongrey_male_0509.jpg') #
    ob.active_material=bpy.data.materials["Material_{}".format(0)]

    return arm_ob,ob

    
def get_minz():
    obj = bpy.context.object.children[0]

    depsgraph = bpy.context.evaluated_depsgraph_get()

    eval_obj_graph=obj.evaluated_get(depsgraph)

    vertices_world = [vertex.co for vertex in eval_obj_graph.data.vertices]

    z_min = (min(vertices_world, key=lambda item: item.z)).z

    return z_min

# def execute(self, context):
#     scene = bpy.context.scene
#     for ob in scene.objects:
#       if ob.type == 'CAMERA':
#         bpy.context.scene.camera = ob
#         print('Set camera %s' % ob.name )
#         file = os.path.join(os.path.dirname(bpy.data.filepath), ob.name )
#         bpy.context.scene.render.filepath = file
#         bpy.ops.render.render( write_still=True )
#     return {'FINISHED'}

def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

# disable render output
def disable_output_start():
    logfile = "/dev/null"
    open(logfile, "a").close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)
    return old

def disable_output_end(old):
    os.close(1)
    os.dup(old)
    os.close(old)

def all_cameras_render_images(root_path,num_frame):
    scene = bpy.context.scene
    for ob in scene.objects:
        if ob.type == 'CAMERA' and ob.name!='Camera':
            path=os.path.join(root_path,ob.name)
            mkdir_safe(path)
            for frame_id in range(num_frame):
                scene.frame_set(frame_id)
                scene.camera = ob
                scene.render.filepath = os.path.join(path,"Image{:04}.png".format(frame_id))
                old = disable_output_start()
                # Render
                bpy.ops.render.render(write_still=True)
                # disable output redirection
                disable_output_end(old)

            

def new_light(r,h):
    # remove default light    
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete(use_global=False)

    r -=1
    locations=[[r,0,h],[-r,0,h],[0,r,h],[0,-r,h]]
    for i in range(4): # new 4 point light
        # Create new light
        lamp_data = bpy.data.lights.new(name="Lamp{}".format(i), type='POINT')
        lamp_data.energy = 1000
        lamp_object = bpy.data.objects.new(name="Lamp{}".format(i), object_data=lamp_data)
        bpy.context.collection.objects.link(lamp_object)
        lamp_object.location = locations[i]
    

def texture_mesh(id,cloth_img_name=None):
    if isinstance(cloth_img_name,type(None)):
        return 
    material = bpy.data.materials.new(name="Material_{}".format(id))
    material.use_nodes = True
    tree=material.node_tree

    # remove all nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
    # uv = tree.nodes.new("ShaderNodeTexCoord")
    # uv.location = -800, 400
    cloth_img = bpy.data.images.load(cloth_img_name)
    uv_im = tree.nodes.new("ShaderNodeTexImage")
    uv_im.location = 400, -400
    uv_im.image = cloth_img

    # normal map
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.location = -400,400

    # PSDF
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled") #ShaderNodeBsdfPrincipled
    bsdf.location = 0,400

    # 
    mat_out = tree.nodes.new("ShaderNodeOutputMaterial")
    mat_out.location=200,200

    # 
    tree.links.new(uv_im.outputs[0],bsdf.inputs[0])
    tree.links.new(normal_map.outputs[0],bsdf.inputs['Normal'])
    tree.links.new(bsdf.outputs[0],mat_out.inputs[0])

    

def Env_hdri(bg_image):
    C = bpy.context
    scn = C.scene
    # Get the environment node tree of the current scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    # Clear all nodes
    tree_nodes.clear()

    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')

    # Add Environment Texture node
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    node_environment.image = bpy.data.images.load(bg_image) # Relative path
    node_environment.location = -300,0

    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
    node_output.location = 200,0

    # Link all nodes
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])



    


model_path='/Volumes/Samsung_T5/data/test_blender/smplx-models/male_smplx.fbx'




# smplx_pose_file='/Volumes/Samsung_T5/dongmi_projects/3D-Tools/vis/data/smplx/smplx_params.json'
# trans_pose,trans_shapes,trans_trans,trans_orient,trans_expression = load_pose_info(smplx_pose_file)


smplx_pose_file='/Volumes/Samsung_T5/data/test_blender/ACCAD/s011/walkdog_stageii.npz'
trans_pose,trans_trans,trans_orient,trans_shapes=load_pose_from_ACCAD(smplx_pose_file)


arm_ob,ob = init_scene(model_path,ground=-0.91)



delta_original=np.array([0,0,0])
# minz=get_minz(ob)
minz=0

for i in range(10):
    print("frame:{}".format(i))
    pose=trans_pose[i]
    shape=trans_shapes[:10]
    t=trans_trans[i]
    r=trans_orient[i]
    # r[:]=0
    # t[:]=0
    # print(pose[:3])
    # pose[:3]=r

    if i==0:
        delta_original=delta_original-t
    
    # print(r)
    # exp=trans_expression[i][0]
    # print(t)

    apply_trans_pose_shape(r,t,pose,shape,arm_ob,ob,delta_original,minz,frame=i)
    # apply_only_shape_get_minz(shape,arm_ob,ob)



# render images
scn=bpy.context.scene
scn.render.resolution_x = 640
scn.render.resolution_y = 480
scn.render.resolution_percentage = 100
scn.cycles.film_transparent = True

root_path='/Volumes/Samsung_T5/data/test_blender/render_output'
num_frame=10
all_cameras_render_images(root_path,num_frame)

