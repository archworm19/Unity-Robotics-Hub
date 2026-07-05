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
    //
    // Wire format, one exchange per simulation step, all values little-endian float32:
    //   Python -> Unity: one value per revolute joint, each in [-1, 1]
    //   Unity -> Python: end-effector position (x, y, z), object position (x, y, z)
    //
    // Physics only advances (via a manual Physics.Simulate call) once per received
    // action, so the simulation is in lockstep with the Python side.
    public class RemoteJointController : MonoBehaviour
    {
        [SerializeField]
        int m_Port = 9000;
        [SerializeField]
        string m_EndEffectorLinkName = "tool_link";
        [SerializeField]
        string m_ObjectName = "Target";

        public float stiffness = 10000f;
        public float damping = 100f;
        public float forceLimit = 1000f;

        ArticulationBody[] m_Joints;
        Transform m_EndEffector;
        Transform m_Object;

        TcpListener m_Listener;
        TcpClient m_Client;
        NetworkStream m_Stream;

        byte[] m_ActionBuffer;
        byte[] m_ObservationBuffer;

        void Start()
        {
            m_Joints = GetComponentsInChildren<ArticulationBody>()
                .Where(joint => joint.jointType == ArticulationJointType.RevoluteJoint)
                .ToArray();

            const float defaultDynamicVal = 10f;
            foreach (var joint in m_Joints)
            {
                joint.jointFriction = defaultDynamicVal;
                joint.angularDamping = defaultDynamicVal;

                var drive = joint.xDrive;
                drive.stiffness = stiffness;
                drive.damping = damping;
                drive.forceLimit = forceLimit;
                joint.xDrive = drive;
            }

            m_EndEffector = GetComponentsInChildren<Transform>()
                .FirstOrDefault(t => t.name == m_EndEffectorLinkName);
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

            m_ActionBuffer = new byte[m_Joints.Length * sizeof(float)];
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
            for (var i = 0; i < m_Joints.Length; i++)
            {
                var action = BitConverter.ToSingle(buffer, i * sizeof(float));
                var drive = m_Joints[i].xDrive;
                var normalized = (action + 1f) * 0.5f; // [-1, 1] -> [0, 1]
                drive.target = Mathf.Lerp(drive.lowerLimit, drive.upperLimit, normalized);
                m_Joints[i].xDrive = drive;
            }
        }

        void WriteObservation(byte[] buffer)
        {
            var endEffectorPosition = m_EndEffector.position;
            var objectPosition = m_Object.position;
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
