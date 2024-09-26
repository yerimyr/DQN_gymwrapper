import gym
from gym import spaces
import simpy
import numpy as np
from environment import  *
from config_SimPy import *

class GymWrapper(gym.Env):
    DAILY_COST_REPORT = {}  # 클래스 변수로 정의

    def __init__(self, I, P):  # GymWrapper 클래스의 인스턴스를 초기화하는 생성자 메소드, 두 개의 매개변수인 I와 P를 받아서 환경을 설정하는 데 사용
        super(GymWrapper, self).__init__()  # 부모 클래스인 gym.Env의 초기화 메소드를 호출하여, Gym 환경에서 필요한 초기 설정을 수행
        self.I = I
        self.P = P
        self.daily_events = []  # 매일 발생하는 이벤트를 저장할 빈 리스트를 초기화
        self.env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.supplierList, self.daily_events = create_env(I, P, self.daily_events)  # create_env 함수를 통해 다양한 환경 구성 요소를 생성하고, 이를 인스턴스 변수로 저장하여 클래스의 다른 메소드나 기능에서 활용할 수 있도록 함

        self.max_order_quantity = 1  # 에이전트는 특정 재료를 0 또는 1 단위만 주문할 수 있음
        self.action_space = spaces.Discrete(2)  # [재료 인덱스, 주문 수량] 형태로 print
        
        self.observation_space = spaces.MultiDiscrete([51, 51])  # total_inventory 공간 51개, on-hand_inventory 공간 51
        
    def reset(self):  # 환경을 초기화하고, 에이전트가 새로운 에피소드를 시작할 때 호출
        self.env = simpy.Environment()  # simpy 라이브러리를 사용하여 새로운 시뮬레이션 환경을 생성
        self.daily_events = []  # 하루 동안 발생할 이벤트를 저장할 빈 리스트를 초기화
        self.env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.supplierList, self.daily_events = create_env(self.I, self.P, self.daily_events)  # create_env 함수를 호출하여 새로운 환경과 그 구성 요소들을 생성
        return self._get_observation()  # 초기 관측 값을 반환하여 에이전트가 새로운 환경에서 시작할 수 있도록 함
    
    def _get_observation(self):  # 현재 재고 상태를 기반으로 관측 값을 생성
        return np.array([int(inventory.on_hand_inventory) for inventory in self.inventoryList], dtype=np.int32)
    
    def step(self, action):  # 에이전트가 취한 행동을 인자로 받아서 그 행동을 환경에 적용하는 메소드
        current_time = self.env.now  # 현재 시뮬레이션 환경의 시간을 current_time 변수에 저장
        target_time = current_time + 24  # 다음 24시간 후의 시간을 target_time으로 설정

        self.env.run(until=target_time)  # target_time까지 시뮬레이션을 실행

        order_quantity = action  # 액션을 주문 수량으로 사용
        material_index = action % len(self.procurementList)
        
        if order_quantity > 0:  # 주문 수량이 0보다 크면 주문을 진행
            procurement = self.procurementList[material_index]  # 해당 인덱스의 조달 옵션을 가져옴
            supplier = self.supplierList[material_index]  # 해당 인덱스의 공급업체를 가져옴
            self.env.process(procurement.order_material(supplier, self.inventoryList[supplier.item_id], self.daily_events))  # 메소드를 호출하여 공급업체에게 재료를 주문

        self.update_daily_report()  # 현재 일일 보고서를 업데이트

        DAILY_COST_REPORT = Cost.cal_cost(instance, cost_type)  ########## environment.py에서 cost 불러오는 함수 아직 구현 미완성 ##########

        total_cost = sum(GymWrapper.DAILY_COST_REPORT.values())  # 모든 비용을 합산하여 총 비용을 계산

        reward = -total_cost  # 비용이 적을수록 보상이 높아지므로, 에이전트는 비용을 최소화하려고 노력

        done = self._check_done_condition()  # 에피소드가 종료되었는지를 확인하는 메소드를 호출

        recent_events = self.daily_events[-5:] if len(self.daily_events) >= 5 else self.daily_events
        print(f"Time: {self.env.now}, Action: {action}, Observation: {self._get_observation()}, Reward: {reward}, Done: {done}, Info: {recent_events}")
        return self._get_observation(), reward, done, {"daily_events": self.daily_events}
    
    def update_daily_report(self):  # 일일 보고서를 업데이트하는 메소드
        print("Updating daily report with inventory:")  # 일일 보고서를 업데이트 중임을 알리는 메시지를 출력
        for inventory in self.inventoryList:  # 재고 목록을 순회
            print(f"{inventory.item_id}: {inventory.on_hand_inventory} on hand")  # 각 재고 항목의 ID와 현재 재고 수준을 출력
                
        if self.daily_events:
            print(f"Recent events: {self.daily_events[-5:]}")  # 이벤트가 있을 경우에만 출력

    def _check_done_condition(self):  # 에피소드가 종료되었는지를 확인하는 메소드
        return self.env.now >= 336  # 현재 시뮬레이션 시간이 336시간(14일) 이상인지를 체크

if __name__ == "__main__":  # 이 파일이 직접 실행될 때만 아래 코드를 실행하도록 함
    I = {
        0: {"ID": 0, "TYPE": "Product", "NAME": "PRODUCT", "CUST_ORDER_CYCLE": 7, "INIT_LEVEL": 0, "DEMAND_QUANTITY": 0, "HOLD_COST": 1, "SETUP_COST_PRO": 1, "DELIVERY_COST": 1, "DUE_DATE": 7, "SHORTAGE_COST_PRO": 50},
        1: {"ID": 1, "TYPE": "Material", "NAME": "MATERIAL 1", "MANU_ORDER_CYCLE": 1, "INIT_LEVEL": 1, "SUP_LEAD_TIME": 2, "HOLD_COST": 1, "PURCHASE_COST": 2, "ORDER_COST_TO_SUP": 1, "LOT_SIZE_ORDER": 0}
    }  # 제품과 재료의 정보를 담고 있는 딕셔너리

    P = {0: {"ID": 0, "PRODUCTION_RATE": 2, "INPUT_TYPE_LIST": [I[1]], "QNTY_FOR_INPUT_ITEM": [1], "OUTPUT": I[0], "PROCESS_COST": 1, "PROCESS_STOP_COST": 2}}  # 생산 프로세스를 정의하는 딕셔너리

    env = GymWrapper(I, P)  # 이전에 정의한 GymWrapper 클래스를 사용하여 새로운 환경 인스턴스를 생성, 이 환경은 제품과 재료의 정보 I와 생산 정보 P를 기반으로 설정
    obs = env.reset()  #  환경을 초기화하고 초기 관측 값을 얻음
    print("Initial Observation:", obs)  # 초기 관측 값을 출력

    done = False
    while not done:  # 에피소드가 종료될 때까지 반복하는 루프
        action = env.action_space.sample()  # 환경의 행동 공간에서 무작위로 행동을 선택
        obs, reward, done, info = env.step(action)  # 선택한 행동을 환경에 적용하고, 새로운 관측 값, 보상, 종료 여부, 추가 정보를 반환받음

    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")
    if info['daily_events']:
        print(f"Info: {info['daily_events'][-3:]}")  # 마지막 3개의 이벤트 출력
    else:
        print("No daily events.")
