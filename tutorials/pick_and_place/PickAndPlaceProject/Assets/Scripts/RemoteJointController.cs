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
    //   Python -> Unity (7 floats): 6 arm-joint targets in [-1, 1], then 1 gripper
    //                    target in [-1, 1] (-1 = fully closed, +1 = fully open)
    //   Unity -> Python (25 floats), all physical/sensed state (not commanded
    //                    values), positions and rotations relative to base_link,
    //                    in Unity's RUF convention:
    //     [0:3]   end-effector position (x, y, z)
    //     [3:7]   end-effector rotation (x, y, z, w quaternion)
    //     [7:8]   gripper width (meters, actual distance between the fingers)
    //     [8:14]  arm joint positions (radians, actual -- joint_1 .. joint_6)
    //     [14:17] goal (TargetPlacement) position (x, y, z)
    //     [17:20] block ("Target") position (x, y, z)
    //     [20:24] block rotation (x, y, z, w quaternion)
    //     [24:25] placement state (float-encoded WirePlacementState below --
    //                    mirrors remote_connection.PlacementState exactly:
    //                    0 = Outside, 1 = InsideFloating, 2 = InsidePlaced,
    //                    3 = FailedFell)
    //
    // Physics only advances (via a manual Physics.Simulate call) once per received
    // action, so the simulation is in lockstep with the Python side.
    //
    // Episodes reset automatically: once the block is placed (InsidePlaced) or
    // falls off the table (FailedFell), that terminal placement state is written
    // to the observation for this tick as usual, and then -- after writing it,
    // so the terminal observation is unaffected -- the block is teleported back
    // to its spawn pose and TargetPlacement's state is cleared. The arm itself is
    // not reset; it carries over into the next episode wherever it was left. So
    // the action received on the following tick is applied to an already-fresh
    // episode and its resulting observation comes back looking like a normal
    // reset state, with no ticks or actions wasted on the reset itself.
    public class RemoteJointController : MonoBehaviour
    {
        const int k_ObservationFloatCount = 25;

        // Mirrors remote_connection.PlacementState on the Python side, exactly.
        enum WirePlacementState
        {
            Outside = 0,
            InsideFloating = 1,
            InsidePlaced = 2,
            FailedFell = 3,
        }

        // How far below its spawn height (meters) the block has to fall before
        // an episode is considered a failure and reset.
        const float k_FallHeightThreshold = 0.2f;

        [SerializeField]
        int m_Port = 9000;
        [SerializeField]
        string m_BaseLinkName = "base_link";
        [SerializeField]
        string m_EndEffectorLinkName = "tool_link";
        [SerializeField]
        string m_ObjectName = "Target"; // the movable block
        [SerializeField]
        string m_GoalName = "TargetPlacement";

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
        Transform m_Object; // the movable block
        Rigidbody m_ObjectRigidbody;
        Vector3 m_ObjectSpawnPosition;
        Quaternion m_ObjectSpawnRotation;
        Transform m_Goal;
        TargetPlacement m_TargetPlacement;

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
                m_ObjectRigidbody = objectGameObject.GetComponent<Rigidbody>();
                m_ObjectSpawnPosition = m_Object.position;
                m_ObjectSpawnRotation = m_Object.rotation;
            }

            var goalGameObject = GameObject.Find(m_GoalName);
            if (goalGameObject == null)
            {
                Debug.LogError($"{nameof(RemoteJointController)} could not find a goal named " +
                    $"'{m_GoalName}' in the scene.");
            }
            else
            {
                m_Goal = goalGameObject.transform;
                m_TargetPlacement = goalGameObject.GetComponent<TargetPlacement>();
                if (m_TargetPlacement == null)
                {
                    Debug.LogError($"{nameof(RemoteJointController)} expected a {nameof(TargetPlacement)} " +
                        $"component on '{m_GoalName}'.");
                }
            }

            m_ActionBuffer = new byte[(m_ArmJoints.Length + 1) * sizeof(float)]; // + 1 for the gripper command
            m_ObservationBuffer = new byte[k_ObservationFloatCount * sizeof(float)];

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

            var placementState = HasObjectFallen()
                ? WirePlacementState.FailedFell
                : (WirePlacementState)(int)m_TargetPlacement.CurrentState;

            WriteObservation(m_ObservationBuffer, placementState);
            m_Stream.Write(m_ObservationBuffer, 0, m_ObservationBuffer.Length);

            if (placementState == WirePlacementState.InsidePlaced || placementState == WirePlacementState.FailedFell)
            {
                ResetEpisode();
            }
        }

        bool HasObjectFallen() => m_Object.position.y < m_ObjectSpawnPosition.y - k_FallHeightThreshold;

        void ResetEpisode()
        {
            m_Object.position = m_ObjectSpawnPosition;
            m_Object.rotation = m_ObjectSpawnRotation;
            if (m_ObjectRigidbody != null)
            {
                m_ObjectRigidbody.linearVelocity = Vector3.zero;
                m_ObjectRigidbody.angularVelocity = Vector3.zero;
            }
            m_TargetPlacement.ResetState();
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

        void WriteObservation(byte[] buffer, WirePlacementState placementState)
        {
            var offset = 0;
            offset = WriteVector3(buffer, offset, m_BaseLink.InverseTransformPoint(m_EndEffector.position));
            offset = WriteQuaternion(buffer, offset, RelativeRotation(m_BaseLink, m_EndEffector));

            var gripperWidth = Vector3.Distance(
                m_GripperJoints[0].transform.position, m_GripperJoints[1].transform.position);
            offset = WriteFloat(buffer, offset, gripperWidth);

            foreach (var joint in m_ArmJoints)
            {
                offset = WriteFloat(buffer, offset, joint.jointPosition[0]);
            }

            offset = WriteVector3(buffer, offset, m_BaseLink.InverseTransformPoint(m_Goal.position));

            offset = WriteVector3(buffer, offset, m_BaseLink.InverseTransformPoint(m_Object.position));
            offset = WriteQuaternion(buffer, offset, RelativeRotation(m_BaseLink, m_Object));

            WriteFloat(buffer, offset, (float)(int)placementState);
        }

        static Quaternion RelativeRotation(Transform reference, Transform target) =>
            Quaternion.Inverse(reference.rotation) * target.rotation;

        static int WriteFloat(byte[] buffer, int offset, float value)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(value), 0, buffer, offset, sizeof(float));
            return offset + sizeof(float);
        }

        static int WriteVector3(byte[] buffer, int offset, Vector3 v)
        {
            offset = WriteFloat(buffer, offset, v.x);
            offset = WriteFloat(buffer, offset, v.y);
            offset = WriteFloat(buffer, offset, v.z);
            return offset;
        }

        static int WriteQuaternion(byte[] buffer, int offset, Quaternion q)
        {
            offset = WriteFloat(buffer, offset, q.x);
            offset = WriteFloat(buffer, offset, q.y);
            offset = WriteFloat(buffer, offset, q.z);
            offset = WriteFloat(buffer, offset, q.w);
            return offset;
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
