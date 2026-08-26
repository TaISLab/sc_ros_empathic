#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist, TwistStamped, Point
from franka_msgs.msg import FrankaState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray

from subject_config import SubjectConfig, SubjectConfigError

class LaneAdmittanceController:
    def __init__(self):
        rospy.init_node('lane_admittance_controller', anonymous=False)

        # =================================================================
        # Configuración del sujeto: TODO lo específico del sujeto sale de
        # subjects/SXX.yaml. Si falta un valor, el nodo aborta con mensaje
        # claro en vez de arrancar con un valor por defecto: un bloque con
        # tau_max=0 daría TL=0 -> Phi=0 -> Gamma=I, es decir, parecería
        # funcionar y sería silenciosamente un baseline.
        # =================================================================
        try:
            cfg = SubjectConfig(rospy.get_param('~subject_file', ''))
        except SubjectConfigError as e:
            rospy.logfatal('subject config: %s', e)
            raise SystemExit(1)
        self.cfg = cfg
        fz = cfg.frozen
        rospy.loginfo('lane_admittance: %s', cfg.summary())

        # --- Gamma: modulación anisótropa de autoridad --------------------
        self.Gamma = np.eye(3)
        self.gamma_stamp = rospy.Time(0)
        self.gamma_timeout = 0.3   # s; más antiguo que esto -> identidad
        rospy.Subscriber('/sc_effort_experiment/gamma', Float64MultiArray,
                         self.gamma_callback, queue_size=1)

        # --- Carga virtual gravitatoria -----------------------------------
        # Fuerza descendente que el sujeto debe sostener para mantener la
        # banda. Es lo que realmente carga el hombro: sin ella el robot
        # sostiene el brazo y no se desarrolla fatiga. El valor lo RESUELVE
        # compute_subject_params.py para que todos los sujetos queden al
        # mismo %MVC. Suele salir pequeño (~1 N) porque el propio peso del
        # brazo ya aporta la mayor parte del par en postura extendida.
        self.f_virtual_load = np.array([0.0, 0.0, cfg.virtual_load_N])

        # Parámetros de la Admitancia (M * a + B * v = F)
        self.M = np.array([1.0, 1.0, 1.0])  # Inercia virtual (kg)
        self.B = np.array([10.0, 10.0, 10.0]) # Amortiguamiento virtual (N*s/m)
        self.v_h = np.zeros(3)  # Velocidad comandada por el humano
        
        # Parámetros del Filtro y Zona Muerta
        self.deadzone = float(fz['deadzone_N'])
        self.alpha_f = 0.1  # Factor del filtro paso bajo (EMA) para la fuerza
        self.f_filtered = np.zeros(3)

        # Parámetros de los Carriles (Virtual Fixtures en Z)
        self.z_carriles = cfg.z_lanes        # escalados por longitud de brazo
        self.z_tol = float(fz['lane_tol_m'])
        # self.Kp_z = 10.0  # Ganancia proporcional
        self.z_activo = self.z_carriles[1] 
        
        # TIPO DE FUNCIÓN DE ATRACCIÓN ('linear', 'exponential' o 'parabolic')
        self.attraction_type = 'parabolic' 
        
        # Ganancias - Lineal
        self.Kp_z_linear = 10.0  
        
        # Ganancias - Exponencial
        self.Kp_z_exp_base = 1.0     
        self.Kp_z_exp_growth = 20.0  
        
        # Ganancias - Parabólica
        # lane_k = 0 mantiene el régimen v_r = 0: la banda es solo visual y
        # el sujeto sostiene la altura. Cualquier valor no nulo es asistencia
        # a la tarea y debe reportarse como tal en el artículo.
        self.Kp_z_para_k = float(fz['lane_k'])
        self.Kp_z_para_d = 0.01     # Zona muerta/libre desde el centro (+- 2 cm)
        
        # LÍMITES DEL ESPACIO DE TRABAJO (Paredes Virtuales)
        # WORKSPACE LIMITS: derived from the subject's lanes, not hardcoded.
        # z_limits used to be a fixed [0.1, 0.7] regardless of the subject
        # file. Raising the seat (recommended to increase the offloading
        # incentive) pushes p_shoulder up, and with it the lane heights
        # (z_lanes = z_shoulder +- span) -- so a fixed 0.7 m ceiling silently
        # clips the upper lane(s): the robot cannot physically reach a target
        # that the task/plot draws as available. Pad the lanes' own range
        # instead of using an unrelated constant.
        if cfg.z_lanes:
            zpad = 0.10
            self.z_limits = [min(cfg.z_lanes) - zpad, max(cfg.z_lanes) + zpad]
        else:
            self.z_limits = [0.1, 0.7]
            rospy.logwarn('no z_lanes in subject file; falling back to '
                          'hardcoded z_limits %s', self.z_limits)
        self.x_limits = [0.2, 0.6]
        self.y_limits = [-0.3, 0.3]
        
        # Parámetros de Seguridad y Suavizado
        self.max_v = 0.2 # m/s
        self.max_accel = 0.4 # Límite de aceleración en m/s^2 (Ajusta este valor)
        self.v_cmd_prev = np.zeros(3) # Para guardar el comando anterior
        
        # Estado actual
        self.p_actual = np.zeros(3) # [x, y, z]
        self.f_ext_raw = np.zeros(3)
        self.first_state_received = False

        # Loop rate
        self.rate_hz = 1000.0
        self.dt = 1.0 / self.rate_hz

        # Publicador y Suscriptor
        self.cmd_pub = rospy.Publisher('/robot_vel_ctrl/vel_cmd', TwistStamped, queue_size=1)
        self.viz_pub = rospy.Publisher('/sc_effort_experiment/visualization', MarkerArray, queue_size=1)
        self.state_sub = rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.state_callback)

    def state_callback(self, msg):
        # Extraer fuerzas externas (O_F_ext_hat_K)
        self.f_ext_raw = np.array([-msg.O_F_ext_hat_K[0], 
                                   -msg.O_F_ext_hat_K[1], 
                                   -msg.O_F_ext_hat_K[2]])
        
        # Extraer posición actual X, Y, Z del efector final
        # O_T_EE es una matriz 4x4 guardada por columnas (column-major)
        # Índices: 12 -> X, 13 -> Y, 14 -> Z
        self.p_actual[0] = msg.O_T_EE[12]
        self.p_actual[1] = msg.O_T_EE[13]
        self.p_actual[2] = msg.O_T_EE[14]
        
        self.first_state_received = True

    def gamma_callback(self, msg):
        if len(msg.data) == 9:
            self.Gamma = np.array(msg.data).reshape(3, 3)
            self.gamma_stamp = rospy.Time.now()

    def apply_deadzone(self, force):
        f_norm = np.linalg.norm(force)
        if f_norm < self.deadzone:
            return np.zeros(3)
        return force * (1.0 - self.deadzone / f_norm)

    def publish_markers(self):
        msg = MarkerArray()
        frame_id = "base_link" # Frame base del sistema
        timestamp = rospy.Time.now()

        # 1. Marcadores de Carriles (Planos semi-transparentes)
        for i, z_carril in enumerate(self.z_carriles):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = timestamp
            m.ns = "carriles"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = (self.x_limits[0] + self.x_limits[1]) / 2.0
            m.pose.position.y = (self.y_limits[0] + self.y_limits[1]) / 2.0
            m.pose.position.z = z_carril
            m.pose.orientation.w = 1.0
            m.scale.x = self.x_limits[1] - self.x_limits[0]
            m.scale.y = self.y_limits[1] - self.y_limits[0]
            m.scale.z = 0.005 # Plano fino
            
            m.color.a = 0.4 # Transparencia
            # Iluminar en verde el carril activo, azul los inactivos
            if abs(self.z_activo - z_carril) < 0.01:
                m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0
            else:
                m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0
            msg.markers.append(m)

        # 2. Marcador de Posición Actual (Esfera Roja)
        m_pos = Marker()
        m_pos.header.frame_id = frame_id
        m_pos.header.stamp = timestamp
        m_pos.ns = "posicion_actual"
        m_pos.id = 10
        m_pos.type = Marker.SPHERE
        m_pos.action = Marker.ADD
        m_pos.pose.position.x = self.p_actual[0]
        m_pos.pose.position.y = self.p_actual[1]
        m_pos.pose.position.z = self.p_actual[2]
        m_pos.pose.orientation.w = 1.0
        m_pos.scale.x = 0.03
        m_pos.scale.y = 0.03
        m_pos.scale.z = 0.03
        m_pos.color.a = 1.0
        m_pos.color.r = 1.0; m_pos.color.g = 0.0; m_pos.color.b = 0.0
        msg.markers.append(m_pos)

        # 3. Marcador de Límites de Espacio de Trabajo (Caja Amarilla de alambre)
        m_box = Marker()
        m_box.header.frame_id = frame_id
        m_box.header.stamp = timestamp
        m_box.ns = "paredes_virtuales"
        m_box.id = 20
        m_box.type = Marker.LINE_LIST
        m_box.action = Marker.ADD
        m_box.pose.orientation.w = 1.0
        m_box.scale.x = 0.005 # Grosor de la línea
        m_box.color.a = 0.8
        m_box.color.r = 1.0; m_box.color.g = 1.0; m_box.color.b = 0.0
        
        # Definir los 8 vértices del prisma
        x1, x2 = self.x_limits
        y1, y2 = self.y_limits
        z1, z2 = self.z_limits
        corners = [
            Point(x1, y1, z1), Point(x2, y1, z1),
            Point(x1, y2, z1), Point(x2, y2, z1),
            Point(x1, y1, z2), Point(x2, y1, z2),
            Point(x1, y2, z2), Point(x2, y2, z2)
        ]
        
        # Conectar los vértices (12 aristas)
        lines = [
            (0,1), (2,3), (0,2), (1,3), # Base inferior
            (4,5), (6,7), (4,6), (5,7), # Base superior
            (0,4), (1,5), (2,6), (3,7)  # Pilares verticales
        ]
        for p1, p2 in lines:
            m_box.points.append(corners[p1])
            m_box.points.append(corners[p2])
            
        msg.markers.append(m_box)

        # Publicar todo el array
        self.viz_pub.publish(msg)

    def control_loop(self):
        rate = rospy.Rate(self.rate_hz)
        
        while not rospy.is_shutdown():
            if not self.first_state_received:
                rate.sleep()
                continue

            # 1. Filtrado de fuerza y deadzone
            f_deadzone = self.apply_deadzone(self.f_ext_raw)
            self.f_filtered = self.alpha_f * f_deadzone + (1.0 - self.alpha_f) * self.f_filtered

            # 2. Dinámica de Admitancia, incluyendo la carga virtual que el
            #    sujeto debe sostener. Se suma DESPUÉS de la zona muerta y del
            #    filtro, para que sea un sesgo constante que la zona muerta no
            #    se pueda comer.
            f_total = self.f_filtered + self.f_virtual_load
            accel = (f_total - self.B * self.v_h) / self.M
            self.v_h += accel * self.dt

            # 3. Lógica Geométrica de Carriles (Eje Z)
            error_z = self.z_activo - self.p_actual[2]
            v_rz = 0.0

            if abs(error_z) <= self.z_tol:
                # Estado 1: Captura
                if self.attraction_type == 'linear':
                    v_rz = self.Kp_z_linear * error_z
                elif self.attraction_type == 'exponential':
                    # np.sign(error_z) asegura que la corrección apunta hacia el centro del carril
                    # (exp(...) - 1) asegura que en el centro exacto la corrección es 0
                    v_rz = np.sign(error_z) * self.Kp_z_exp_base * (np.exp(self.Kp_z_exp_growth * abs(error_z)) - 1.0)
                elif self.attraction_type == 'parabolic':
                    # np.maximum evalúa a 0 si el error es menor que la zona muerta (d)
                    deadband_error = max(0.0, abs(error_z) - self.Kp_z_para_d)
                    v_rz = np.sign(error_z) * self.Kp_z_para_k * (deadband_error ** 2)
            else:
                # Estado 2 & 3: Movimiento Libre y Enganche
                for z_nuevo in self.z_carriles:
                    if abs(z_nuevo - self.p_actual[2]) <= self.z_tol:
                        self.z_activo = z_nuevo
                        break

            # 4. Fusión de comandos, aplicando la modulación anisótropa SOLO
            #    al comando humano.
            #    Fail-safe: si Gamma llega vieja o no llega, se vuelve a la
            #    identidad, de modo que una caída de sc_fatigue_node degrada a
            #    admitancia pura en lugar de congelar una atenuación arbitraria
            #    sobre el sujeto.
            if (rospy.Time.now() - self.gamma_stamp).to_sec() > self.gamma_timeout:
                G = np.eye(3)
            else:
                G = self.Gamma

            v_cmd = G.dot(self.v_h)
            v_cmd[2] += v_rz # Añadir atracción del carril

            # ---------------------------------------------------------
            # 5. LÓGICA DE PAREDES VIRTUALES (Workspace limits)
            # Verificamos si estamos fuera de los límites y empujando hacia afuera.
            # Si es así, se anula el comando Y se vacía el integrador de admitancia.
            
            # Eje X
            if self.p_actual[0] <= self.x_limits[0] and v_cmd[0] < 0:
                v_cmd[0] = 0.0
                self.v_h[0] = 0.0
            elif self.p_actual[0] >= self.x_limits[1] and v_cmd[0] > 0:
                v_cmd[0] = 0.0
                self.v_h[0] = 0.0

            # Eje Y
            if self.p_actual[1] <= self.y_limits[0] and v_cmd[1] < 0:
                v_cmd[1] = 0.0
                self.v_h[1] = 0.0
            elif self.p_actual[1] >= self.y_limits[1] and v_cmd[1] > 0:
                v_cmd[1] = 0.0
                self.v_h[1] = 0.0

            # Eje Z (Nota: Los carriles están dentro de estos límites, así que conviven bien)
            if self.p_actual[2] <= self.z_limits[0] and v_cmd[2] < 0:
                v_cmd[2] = 0.0
                self.v_h[2] = 0.0
            elif self.p_actual[2] >= self.z_limits[1] and v_cmd[2] > 0:
                v_cmd[2] = 0.0
                self.v_h[2] = 0.0
            # ---------------------------------------------------------

            # 6. Saturación de magnitud (Seguridad general)
            v_norm = np.linalg.norm(v_cmd)
            if v_norm > self.max_v:
                v_cmd = v_cmd * (self.max_v / v_norm)

            # ---------------------------------------------------------
            # 7. Saturación de magnitud de aceleración (Rate Limiter)
            dv = v_cmd - self.v_cmd_prev
            dv_norm = np.linalg.norm(dv)
            max_dv = self.max_accel * self.dt
            
            if dv_norm > max_dv:
                v_cmd = self.v_cmd_prev + (dv / dv_norm) * max_dv
                
            self.v_cmd_prev = np.copy(v_cmd)
            
            # Sincronizar el integrador de admitancia para evitar windup.
            # IMPORTANTE: hay que resincronizar en el espacio SIN MODULAR. Si
            # se realimenta el comando ya modulado, Gamma se vuelve a aplicar
            # en el ciclo siguiente y la atenuación se compone hasta que la
            # interfaz se siente muerta. Gamma se mantiene definida positiva
            # (suelo de autovalor en sc_fatigue_node) precisamente para que
            # esta inversa exista siempre y esté bien condicionada.
            v_exec_h = np.copy(v_cmd)
            v_exec_h[2] -= v_rz
            try:
                self.v_h = np.linalg.solve(G, v_exec_h)
            except np.linalg.LinAlgError:
                self.v_h = v_exec_h
            # ---------------------------------------------------------

            # 8. Publicar Comandos al Robot
            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time.now()
            twist_msg.twist.linear.x = v_cmd[0]
            twist_msg.twist.linear.y = v_cmd[1]
            twist_msg.twist.linear.z = v_cmd[2]
            # Las velocidades angulares se mantienen en 0 (puedes añadir admitancia orientacional si lo necesitas)
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.0

            self.cmd_pub.publish(twist_msg)
            
            # 8. Publicar visualizaciones a RViz
            self.publish_markers()
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = LaneAdmittanceController()
        rospy.loginfo("Iniciando nodo de control de admitancia por carriles...")
        controller.control_loop()
    except rospy.ROSInterruptException:
        pass