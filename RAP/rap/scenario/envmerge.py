from typing import List, Tuple, Union, Dict, Optional
from datetime import datetime
import math
import os

from highway_env.road.road import Road, RoadNetwork, LaneIndex
from highway_env.road.lane import StraightLane
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.behavior import IDMVehicle
import numpy as np

from rap.scenario.DBBridge import DBBridge
from rap.scenario.envPlotter import ScePlotter

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'Turn-left - change lane to the left of the current lane',
    1: 'IDLE - remain in the current lane with current speed',
    2: 'Turn-right - change lane to the right of the current lane',
    3: 'Acceleration - accelerate the vehicle',
    4: 'Deceleration - decelerate the vehicle'
}


class EnvScenario:
    """
    场景描述器（兼容原接口），增强适配 merge-v0（车道汇入）：
    - 支持非 StraightLane 的纵向位置计算（local_coordinates）
    - 非 junction 下也描述周车（merge 决策关键）
    - 输出每个车道(当前/左/右)最近前车+后车，便于 LLM 推理/与 TTC 对齐
    """

    def __init__(
        self,
        env: AbstractEnv,
        envType: str,
        seed: int,
        database: str = None
    ) -> None:
        self.env = env
        self.envType = envType
        self.ego: MDPVehicle = env.vehicle

        # 这几个参数看起来是你之前用于某些几何计算的常量，这里保留不动，避免不兼容
        self.theta1 = math.atan(3 / 17.5)
        self.theta2 = math.atan(2 / 2.5)
        self.radius1 = np.linalg.norm([3, 17.5])
        self.radius2 = np.linalg.norm([2, 2.5])

        self.road: Road = env.road
        self.network: RoadNetwork = self.road.network
        self.plotter = ScePlotter()

        if database:
            self.database = database
        else:
            self.database = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S') + '.db'

        if os.path.exists(self.database):
            os.remove(self.database)

        self.dbBridge = DBBridge(self.database, env)
        self.dbBridge.createTable()
        self.dbBridge.insertSimINFO(envType, seed)
        self.dbBridge.insertNetwork()

    # =========================
    # 基础工具函数（增强鲁棒性）
    # =========================
    def _safe_lane_id(self, lane_index: LaneIndex) -> Optional[int]:
        """
        lane_index 通常是 (start_node, end_node, lane_id)
        但不同版本/环境可能略有差异，这里做鲁棒处理。
        """
        try:
            if lane_index is None:
                return None
            if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 3:
                return int(lane_index[2])
        except Exception:
            pass
        return None

    def _safe_road_key(self, lane_index: LaneIndex) -> Optional[Tuple]:
        """
        用于区分“主路/匝道”等 road segment：
        取 lane_index 的前两项 (start, end) 作为 road key。
        """
        try:
            if lane_index is None:
                return None
            if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 2:
                return (lane_index[0], lane_index[1])
        except Exception:
            pass
        return None

    def _get_lane(self, lane_index: LaneIndex):
        """安全获取 lane 对象"""
        try:
            return self.network.get_lane(lane_index)
        except Exception:
            return None

    def _local_coordinates(self, lane, position: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        通用 local_coordinates：
        highway-env 的 lane 大多支持 local_coordinates -> (s, d)
        """
        try:
            if lane is None:
                return None
            s, d = lane.local_coordinates(position)
            return float(s), float(d)
        except Exception:
            return None

    # =========================
    # 周车检索与绘图（保持接口）
    # =========================
    def getSurrendVehicles(self, vehicles_count: int) -> List[IDMVehicle]:
        """
        保持原接口：返回靠近自车的车辆列表（不含自车）
        """
        return self.road.close_vehicles_to(
            self.ego,
            self.env.PERCEPTION_DISTANCE,
            count=vehicles_count - 1,
            see_behind=True,
            sort='sorted'
        )

    def plotSce(self, fileName: str) -> None:
        SVs = self.getSurrendVehicles(10)
        self.plotter.plotSce(self.network, SVs, self.ego, fileName)

    def getUnitVector(self, radian: float) -> Tuple[float]:
        return (math.cos(radian), math.sin(radian))

    # =========================
    # junction 判断（保持原逻辑）
    # =========================
    def isInJunction(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> bool:
        if self.envType == 'intersection-v1':
            x, y = vehicle.position
            return -20 <= x <= 20 and -20 <= y <= 20
        else:
            return False

    # =========================
    # 纵向位置计算（merge 场景必须增强）
    # =========================
    def getLanePosition(self, vehicle: Union[IDMVehicle, MDPVehicle]) -> Union[float, None]:
        """
        原实现只支持 StraightLane，会导致 merge-v0 下经常拿不到 lane_position。
        新实现：
        - 优先用 lane.local_coordinates(position) -> s 作为纵向位置（几乎适配所有 lane 类型）
        - 若失败再尝试 StraightLane 的 start 距离兜底
        """
        if self.isInJunction(vehicle):
            return None

        currentLaneIdx = getattr(vehicle, "lane_index", None)
        lane = self._get_lane(currentLaneIdx)
        if lane is None:
            return None

        # 1) 通用方式：local_coordinates
        lcd = self._local_coordinates(lane, vehicle.position)
        if lcd is not None:
            s, _ = lcd
            return float(s)

        # 2) 兜底：若是 StraightLane，用 start 距离
        if isinstance(lane, StraightLane):
            try:
                return float(np.linalg.norm(vehicle.position - lane.start))
            except Exception:
                return None

        return None

    # =========================
    # 动作可用性描述（接口保持）
    # =========================
    def availableActionsDescription(self) -> str:
        """
        兼容原接口：
        - intersection 中仍禁用变道（原逻辑保留）
        - merge-v0 中不禁用动作，只在文字上提示“合流区域变道风险高”
        """
        description = 'Your available actions are: \n'
        availableActions = self.env.get_available_actions()

        # merge-v0 文本提示（不改变动作集合，避免和外部策略不兼容）
        if self.envType == 'merge-v0':
            description = (
                "Your available actions are (MERGE AREA NOTICE):\n"
                "- This is a merging/ramp-insertion scenario. Lane changes require extra caution, "
                "especially checking vehicles BEHIND in the target lane (rear-collision risk).\n"
            )

        for action in availableActions:
            # junction 内禁用 LANE_LEFT / LANE_RIGHT（原逻辑保留）
            if self.isInJunction(self.ego) and action in [0, 2]:
                continue
            description += ACTIONS_DESCRIPTION[action] + f' Action_id: {action}\n'
        return description

    # =========================
    # 自车道路信息描述（增强 merge 语义）
    # =========================
    def processNormalLane(self, lidx: LaneIndex) -> str:
        """
        注意：merge 场景中车道数量会随 road segment 变化。
        这里描述“当前 road segment 的车道数”，避免误导。
        """
        sideLanes = self.network.all_side_lanes(lidx)
        numLanes = len(sideLanes)

        if numLanes <= 1:
            description = "You are driving on the current road segment with only one lane, lane changes may be limited. "
        else:
            egoLaneRank = self._safe_lane_id(lidx)
            if egoLaneRank is None:
                description = f"You are driving on the current road segment with {numLanes} lanes. "
            else:
                if egoLaneRank == 0:
                    description = f"You are driving on the current road segment with {numLanes} lanes, and you are currently driving in the leftmost lane of this segment. "
                elif egoLaneRank == numLanes - 1:
                    description = f"You are driving on the current road segment with {numLanes} lanes, and you are currently driving in the rightmost lane of this segment. "
                else:
                    laneRankDict = {1: 'second', 2: 'third', 3: 'fourth'}
                    rank_word = laneRankDict.get(egoLaneRank, f"{egoLaneRank + 1}th")
                    description = f"You are driving on the current road segment with {numLanes} lanes, and you are currently driving in the {rank_word} lane from the left (within this segment). "

        # merge-v0 的额外提醒（不影响其它 env）
        if self.envType == 'merge-v0':
            description += "Note: this is a merging area; lane layout, priorities, and safe gaps may change quickly. "

        lane_pos = self.getLanePosition(self.ego)
        if lane_pos is not None:
            description += (
                f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, "
                f"speed is {self.ego.speed:.2f} m/s, acceleration is {self.ego.action['acceleration']:.2f} m/s^2, "
                f"and longitudinal lane position is {lane_pos:.2f} m.\n"
            )
        else:
            description += (
                f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, "
                f"speed is {self.ego.speed:.2f} m/s, acceleration is {self.ego.action['acceleration']:.2f} m/s^2.\n"
            )

        return description

    # =========================
    # 周车相对状态（增强 merge 可读性）
    # =========================
    def getSVRelativeState(self, sv: IDMVehicle) -> str:
        """
        原版只输出 ahead/behind。
        这里保留 ahead/behind 逻辑，但在 merge 下补充“left/right/offset”信息（不破坏文本结构）。
        """
        relativePosition = sv.position - self.ego.position
        egoUnitVector = self.getUnitVector(self.ego.heading)
        cosineValue = sum([x * y for x, y in zip(relativePosition, egoUnitVector)])
        ahead_behind = 'ahead of you' if cosineValue >= 0 else 'behind you'

        # merge 场景补充横向信息（不强依赖 lane 类型）
        if self.envType == 'merge-v0':
            try:
                # 用“自车朝向的法向量”判断左右（近似）
                normal = np.array([-egoUnitVector[1], egoUnitVector[0]], dtype=float)
                lateral = float(np.dot(relativePosition, normal))
                if lateral > 1.0:
                    side = "to your left"
                elif lateral < -1.0:
                    side = "to your right"
                else:
                    side = "near your lane"
                return f"{ahead_behind} and {side}"
            except Exception:
                return ahead_behind

        return ahead_behind

    # =========================
    # junction 周车描述（保持原接口）
    # =========================
    def describeSVJunctionLane(self, currentLaneIndex: LaneIndex) -> str:
        surroundVehicles = self.getSurrendVehicles(6)
        if not surroundVehicles:
            return "There are no other vehicles driving near you, you can drive freely.\n"
        SVDescription = ''
        for sv in surroundVehicles:
            SVDescription += (
                f"- Vehicle `{id(sv) % 1000}` is {self.getSVRelativeState(sv)}. "
                f"Position `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, "
                f"speed {sv.speed:.2f} m/s, acceleration {sv.action['acceleration']:.2f} m/s^2.\n"
            )
        return "Other vehicles near you:\n" + SVDescription

    # =========================
    # 非 junction 周车描述（merge 关键增强）
    # =========================
    def _pick_front_rear_in_lane(
        self,
        lane_index: LaneIndex,
        vehicles: List[IDMVehicle],
        ego_s: float
    ) -> Dict[str, Optional[IDMVehicle]]:
        """
        在某条 lane_index 对应车道里，挑最近前车/后车（基于纵向 s）。
        返回 {"front": vehicle or None, "rear": vehicle or None}
        """
        lane = self._get_lane(lane_index)
        if lane is None:
            return {"front": None, "rear": None}

        best_front = None
        best_front_ds = float("inf")
        best_rear = None
        best_rear_ds = float("inf")

        for v in vehicles:
            if getattr(v, "lane_index", None) is None:
                continue
            # 车道匹配：lane_index 精确匹配更稳（merge 多 road segment）
            if v.lane_index != lane_index:
                continue

            lcd = self._local_coordinates(lane, v.position)
            if lcd is None:
                continue
            s_v, _ = lcd
            ds = s_v - ego_s
            if ds >= 0:
                if ds < best_front_ds:
                    best_front_ds = ds
                    best_front = v
            else:
                if -ds < best_rear_ds:
                    best_rear_ds = -ds
                    best_rear = v

        return {"front": best_front, "rear": best_rear}

    def describeSVNormalLane(self, currentLaneIndex: LaneIndex, surroundVehicles: List[IDMVehicle]) -> str:
        """
        merge-v0 决策最需要的信息：当前车道 + 左右车道 的最近前车/后车。
        这里输出结构化文本，保证 LLM 能“看懂 gap”与 rear 风险。
        """
        if not surroundVehicles:
            return "There are no other vehicles driving near you, you can drive freely.\n"

        lane = self._get_lane(currentLaneIndex)
        ego_lcd = self._local_coordinates(lane, self.ego.position) if lane is not None else None
        if ego_lcd is None:
            # 兜底：没有 ego_s 就退化成简单列表
            SVDescription = ''
            for sv in surroundVehicles:
                SVDescription += (
                    f"- Vehicle `{id(sv) % 1000}` is {self.getSVRelativeState(sv)}. "
                    f"Position `({sv.position[0]:.2f}, {sv.position[1]:.2f})`, "
                    f"speed {sv.speed:.2f} m/s, acceleration {sv.action['acceleration']:.2f} m/s^2.\n"
                )
            return "Other vehicles near you:\n" + SVDescription

        ego_s, _ = ego_lcd

        # 目标车道：当前、左、右（在当前 road segment 里）
        sideLanes = self.network.all_side_lanes(currentLaneIndex)
        # 找到当前 lane 在 sideLanes 中的序号
        cur_lane_id = self._safe_lane_id(currentLaneIndex)
        # 如果取不到 lane_id，就只描述当前车道
        lane_indices_to_check: Dict[str, Optional[LaneIndex]] = {"current": currentLaneIndex, "left": None, "right": None}

        if cur_lane_id is not None and sideLanes:
            # sideLanes 通常是 LaneIndex 列表，lane_id 就是第三位
            # 当前 lane_id - 1 是左，+1 是右（在同一 road segment）
            left_lane = None
            right_lane = None
            for l in sideLanes:
                lid = self._safe_lane_id(l)
                if lid == cur_lane_id - 1:
                    left_lane = l
                if lid == cur_lane_id + 1:
                    right_lane = l
            lane_indices_to_check["left"] = left_lane
            lane_indices_to_check["right"] = right_lane

        # 在三个车道上分别挑 front/rear
        picked = {}
        for k, lidx in lane_indices_to_check.items():
            if lidx is None:
                picked[k] = {"front": None, "rear": None, "lane_index": None}
            else:
                fr = self._pick_front_rear_in_lane(lidx, surroundVehicles, ego_s)
                picked[k] = {"front": fr["front"], "rear": fr["rear"], "lane_index": lidx}

        # merge 风险提示：如果周车 road key 与 ego 不同，说明可能来自匝道/不同路段
        ego_road_key = self._safe_road_key(self.ego.lane_index)
        has_other_road_vehicle = False
        for sv in surroundVehicles:
            if self._safe_road_key(getattr(sv, "lane_index", None)) != ego_road_key:
                has_other_road_vehicle = True
                break

        out = "Other vehicles near you (closest in each lane):\n"
        if self.envType == "merge-v0" and has_other_road_vehicle:
            out += "- Notice: vehicles from different road segments detected (possible ramp / merge traffic). Rear checks are critical.\n"

        def _veh_line(prefix: str, veh: Optional[IDMVehicle], lane_tag: str) -> str:
            if veh is None:
                return f"- {lane_tag} {prefix}: None\n"
            lp = self.getLanePosition(veh)
            lp_str = f"{lp:.2f} m" if lp is not None else "N/A"
            return (
                f"- {lane_tag} {prefix}: Vehicle `{id(veh) % 1000}` is {self.getSVRelativeState(veh)}. "
                f"Position `({veh.position[0]:.2f}, {veh.position[1]:.2f})`, "
                f"speed {veh.speed:.2f} m/s, acceleration {veh.action['acceleration']:.2f} m/s^2, "
                f"lane position {lp_str}.\n"
            )

        out += _veh_line("Front", picked["current"]["front"], "Current lane")
        out += _veh_line("Rear", picked["current"]["rear"], "Current lane")

        # 左右车道存在时才输出（避免误导）
        if picked["left"]["lane_index"] is not None:
            out += _veh_line("Front", picked["left"]["front"], "Left lane")
            out += _veh_line("Rear", picked["left"]["rear"], "Left lane")
        if picked["right"]["lane_index"] is not None:
            out += _veh_line("Front", picked["right"]["front"], "Right lane")
            out += _veh_line("Rear", picked["right"]["rear"], "Right lane")

        return out

    # =========================
    # 总描述（接口保持，merge 下增强内容）
    # =========================
    def describe(self, decisionFrame: int) -> str:
        """
        保持原接口：返回字符串描述。
        关键增强：
        - 非 junction 情况不再清空 SVDescription
        - merge-v0 下输出“当前/左/右车道最近前后车”
        """
        surroundVehicles = self.getSurrendVehicles(10)
        self.dbBridge.insertVehicle(decisionFrame, surroundVehicles)
        currentLaneIndex = self.ego.lane_index

        if self.isInJunction(self.ego):
            roadCondition = (
                f"You are driving in an intersection, you can't change lane. "
                f"Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, "
                f"speed {self.ego.speed:.2f} m/s, acceleration {self.ego.action['acceleration']:.2f} m/s^2.\n"
            )
            SVDescription = self.describeSVJunctionLane(currentLaneIndex)
        else:
            roadCondition = self.processNormalLane(currentLaneIndex)

            # ✅ merge/普通高速场景都输出周车信息（对 LLM 和记忆检索都更友好）
            SVDescription = self.describeSVNormalLane(currentLaneIndex, surroundVehicles)

        return roadCondition + SVDescription

    # =========================
    # DB 提交（保持接口）
    # =========================
    def promptsCommit(
        self,
        decisionFrame: int,
        vectorID: str,
        done: bool,
        description: str,
        fewshots: str,
        thoughtsAndAction: str
    ):
        self.dbBridge.insertPrompts(
            decisionFrame, vectorID, done, description,
            fewshots, thoughtsAndAction
        )
