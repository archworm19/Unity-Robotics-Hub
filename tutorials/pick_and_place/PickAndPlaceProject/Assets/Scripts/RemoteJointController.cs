using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

namespace Unity.Robotics.PickAndPlace
{
    // Alternative to the keyboard-driven Controller (from the URDF Importer package):
    // joint targets come from a Python process over a raw TCP socket instead of
    // arrow-key input. Only one of the two should drive the robot's joints at a time.
    // Unity is fully agnostic to user input -- it only ever executes the 7-DOF
    // signal it's given; any keyboard/UI handling lives entirely on the Python side.
    //
    // Wire format, one exchange per simulation step, all values little-endian float32:
    //   Python -> Unity: 6 arm-joint targets in [-1, 1], then 1 gripper target in
    //                    [-1, 1] (-1 = fully closed, +1 = fully open)
    //   Unity -> Python: end-effector position (x, y, z), object position (x, y, z),
    //                    both relative to base_link, in Unity's RUF convention
    //
    // Physics only advances (via a manual Physics.Simulate call) once per received
    // action, so the simulation is in lockstep with the Python side.
    public class RemoteJointController : MonoBehaviour
    {
        [SerializeField]
        int m_Port = 9000;
        [SerializeField]
        string m_BaseLinkName = "base_link";
        [SerializeField]
        string m_EndEffectorLinkName = "tool_link";
        [SerializeField]
        string m_ObjectName = "Target";

        public float stiffness = 10000f;
        public float damping = 100f;
        public float forceLimit = 1000f;

        // niryo_one.urdf declares gripper_joint_right as
        // <mimic joint="gripper_joint_left" multiplier="-1"/>, i.e. the two
        // fingers are meant to move in mirrored directions. URDF-Importer
        // doesn't implement <mimic>, so that coupling is reproduced by hand
        // here via a per-joint sign (both joints share the same, symmetric-
        // around-zero limits, so negating the command is exactly equivalent
        // to the multiplier="-1" relationship).
        const string k_RightGripperLinkName = "right_gripper";

        ArticulationBody[] m_ArmJoints;
        ArticulationBody[] m_GripperJoints;
        float[] m_GripperJointSigns;
        Transform m_BaseLink;
        Transform m_EndEffector;
        Transform m_Object;

        TcpListener m_Listener;
        TcpClient m_Client;
        NetworkStream m_Stream;

        byte[] m_ActionBuffer;
        byte[] m_ObservationBuffer;

        void Start()
        {
            m_ArmJoints = GetComponentsInChildren<ArticulationBody>()
                .Where(joint => joint.jointType == ArticulationJointType.RevoluteJoint)
                .ToArray();
            m_GripperJoints = GetComponentsInChildren<ArticulationBody>()
                .Where(joint => joint.jointType == ArticulationJointType.PrismaticJoint)
                .ToArray();
            m_GripperJointSigns = m_GripperJoints
                .Select(joint => joint.name == k_RightGripperLinkName ? -1f : 1f)
                .ToArray();

            const float defaultDynamicVal = 10f;
            foreach (var joint in m_ArmJoints.Concat(m_GripperJoints))
            {
                joint.jointFriction = defaultDynamicVal;
                joint.angularDamping = defaultDynamicVal;

                var drive = joint.xDrive;
                drive.stiffness = stiffness;
                drive.damping = damping;
                drive.forceLimit = forceLimit;
                joint.xDrive = drive;
            }

            var childTransforms = GetComponentsInChildren<Transform>();

            m_BaseLink = childTransforms.FirstOrDefault(t => t.name == m_BaseLinkName);
            if (m_BaseLink == null)
            {
                Debug.LogError($"{nameof(RemoteJointController)} could not find a link named " +
                    $"'{m_BaseLinkName}' under {name}.");
            }

            m_EndEffector = childTransforms.FirstOrDefault(t => t.name == m_EndEffectorLinkName);
            if (m_EndEffector == null)
            {
                Debug.LogError($"{nameof(RemoteJointController)} could not find a link named " +
                    $"'{m_EndEffectorLinkName}' under {name}.");
            }

            var objectGameObject = GameObject.Find(m_ObjectName);
            if (objectGameObject == null)
            {
                Debug.LogError($"{nameof(RemoteJointController)} could not find an object named " +
                    $"'{m_ObjectName}' in the scene.");
            }
            else
            {
                m_Object = objectGameObject.transform;
            }

            m_ActionBuffer = new byte[(m_ArmJoints.Length + 1) * sizeof(float)]; // + 1 for the gripper command
            m_ObservationBuffer = new byte[6 * sizeof(float)];

            Physics.autoSimulation = false;

            m_Listener = new TcpListener(IPAddress.Any, m_Port);
            m_Listener.Start();
            Debug.Log($"{nameof(RemoteJointController)} listening on port {m_Port}.");
        }

        void Update()
        {
            if (m_Client == null)
            {
                if (m_Listener.Pending())
                {
                    m_Client = m_Listener.AcceptTcpClient();
                    m_Client.NoDelay = true;
                    m_Stream = m_Client.GetStream();
                    Debug.Log($"{nameof(RemoteJointController)} client connected.");
                }
                return;
            }

            if (!m_Stream.DataAvailable)
            {
                return;
            }

            ReadExact(m_Stream, m_ActionBuffer);
            ApplyActions(m_ActionBuffer);

            Physics.Simulate(Time.fixedDeltaTime);

            WriteObservation(m_ObservationBuffer);
            m_Stream.Write(m_ObservationBuffer, 0, m_ObservationBuffer.Length);
        }

        void ApplyActions(byte[] buffer)
        {
            for (var i = 0; i < m_ArmJoints.Length; i++)
            {
                var action = BitConverter.ToSingle(buffer, i * sizeof(float));
                ApplyNormalizedTarget(m_ArmJoints[i], action);
            }

            var gripperCommand = BitConverter.ToSingle(buffer, m_ArmJoints.Length * sizeof(float));
            for (var i = 0; i < m_GripperJoints.Length; i++)
            {
                ApplyNormalizedTarget(m_GripperJoints[i], gripperCommand * m_GripperJointSigns[i]);
            }
        }

        static void ApplyNormalizedTarget(ArticulationBody joint, float normalizedTarget)
        {
            var drive = joint.xDrive;
            var normalized = (normalizedTarget + 1f) * 0.5f; // [-1, 1] -> [0, 1]
            drive.target = Mathf.Lerp(drive.lowerLimit, drive.upperLimit, normalized);
            joint.xDrive = drive;
        }

        void WriteObservation(byte[] buffer)
        {
            var endEffectorPosition = m_BaseLink.InverseTransformPoint(m_EndEffector.position);
            var objectPosition = m_BaseLink.InverseTransformPoint(m_Object.position);
            Buffer.BlockCopy(BitConverter.GetBytes(endEffectorPosition.x), 0, buffer, 0, sizeof(float));
            Buffer.BlockCopy(BitConverter.GetBytes(endEffectorPosition.y), 0, buffer, 4, sizeof(float));
            Buffer.BlockCopy(BitConverter.GetBytes(endEffectorPosition.z), 0, buffer, 8, sizeof(float));
            Buffer.BlockCopy(BitConverter.GetBytes(objectPosition.x), 0, buffer, 12, sizeof(float));
            Buffer.BlockCopy(BitConverter.GetBytes(objectPosition.y), 0, buffer, 16, sizeof(float));
            Buffer.BlockCopy(BitConverter.GetBytes(objectPosition.z), 0, buffer, 20, sizeof(float));
        }

        static void ReadExact(NetworkStream stream, byte[] buffer)
        {
            var offset = 0;
            while (offset < buffer.Length)
            {
                var read = stream.Read(buffer, offset, buffer.Length - offset);
                if (read == 0)
                {
                    throw new IOException("Socket closed while reading.");
                }
                offset += read;
            }
        }

        void OnApplicationQuit()
        {
            m_Stream?.Close();
            m_Client?.Close();
            m_Listener?.Stop();
        }
    }
}
