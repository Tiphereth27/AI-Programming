"""
SARSA 강화학습 기반 시간표 추천기
"""
import os
import json
import logging
import random
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set
from ..base import Course, UserPreferences
from .base_recommender import BaseRecommender, State, Action, Reward
from utils.html_generator import generate_html_timetable

class SarsaRecommender(BaseRecommender):
    """SARSA 강화학습 기반 시간표 추천기"""
    
    def __init__(self, courses: List[Course], preferences: Optional[UserPreferences] = None):
        """
        Args:
            courses (List[Course]): 추천할 수 있는 과목 목록
            preferences (Optional[UserPreferences]): 사용자 선호도 설정
        """
        super().__init__(courses, preferences)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # SARSA 관련 변수
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
        # 과목 유형별 가중치
        self.course_type_weights = {
            "전공필수": 1.5,
            "전공선택": 1.2,
        }
        
        # 학년별 가중치
        self.year_weights = {
            1: 2.0,
            2: 2.0,
            3: 2.0,
            4: 2.0
        }

        # 저장된 시간표
        self.saved_schedules = {}

        # 학습 설정
        self.training_config = {
            'num_episodes': 1000,
            'max_steps_per_episode': 20,
            'epsilon_decay': 0.995,
            'min_epsilon': 0.01,
            'target_reward': 100.0
        }

    def _get_state_key(self, state: State) -> str:
        """
        상태를 문자열 키로 변환합니다.
        
        Args:
            state (State): 현재 상태
            
        Returns:
            str: 상태 키
        """
        try:
            # 상태의 주요 특징들을 문자열로 변환
            state_parts = [
                f"credits_{state.total_credits}",
                f"courses_{len(state.selected_courses)}",
                f"types_{'-'.join(f'{k}:{v}' for k, v in state.course_types.items())}",
                f"years_{'-'.join(f'{k}:{v}' for k, v in state.years.items())}"
            ]
            return "_".join(state_parts)
            
        except Exception as e:
            self.logger.error(f"상태 키 생성 중 오류 발생: {str(e)}")
            return "error_state"

    def _get_action_key(self, action: Action) -> str:
        """
        행동을 문자열 키로 변환합니다.
        
        Args:
            action (Action): 행동
            
        Returns:
            str: 행동 키
        """
        try:
            if action.time_slot:
                day, start, end = action.time_slot
                return f"{action.course.code}_{day}_{start}_{end}"
            return action.course.code
            
        except Exception as e:
            self.logger.error(f"행동 키 생성 중 오류 발생: {str(e)}")
            return "error_action"

    def _select_action(self, state: State, available_actions: List[Action]) -> Optional[Action]:
        """
        ε-greedy 정책에 따라 행동을 선택합니다.
        
        Args:
            state (State): 현재 상태
            available_actions (List[Action]): 가능한 행동 목록
            
        Returns:
            Optional[Action]: 선택된 행동
        """
        try:
            if not available_actions:
                return None
                
            state_key = self._get_state_key(state)
            
            # 탐험 (랜덤 선택)
            if random.random() < self.epsilon:
                return random.choice(available_actions)
                
            # 활용 (Q-value 기반 선택)
            q_values = [
                (action, self.q_table[state_key][self._get_action_key(action)])
                for action in available_actions
            ]
            max_q = max(q_values, key=lambda x: x[1])[1]
            best_actions = [action for action, q in q_values if q == max_q]
            return random.choice(best_actions)
            
        except Exception as e:
            self.logger.error(f"행동 선택 중 오류 발생: {str(e)}")
            return random.choice(available_actions) if available_actions else None

    def _calculate_reward(self, state: State, action: Action) -> float:
        """
        행동에 대한 보상을 계산합니다.
        
        Args:
            state (State): 현재 상태
            action (Action): 선택된 행동
            
        Returns:
            float: 계산된 보상
        """
        try:
            if not state.selected_courses:
                return 0.0
                
            reward = 0.0
            
            # 시간 충돌 체크
            if any(self.check_time_conflict(action.course, course) for course in state.selected_courses):
                return -10.0
                
            # 기본 점수 계산
            reward = self._calculate_course_score(action.course)
            
            # 시간대 선호도 추가 보상
            for slot in action.course.time_slots:
                if slot.day in self.preferences.preferred_days:
                    reward += 0.5
                if any(time_range.day == slot.day and
                      time_range.start_time <= slot.start_time and 
                      slot.end_time <= time_range.end_time
                      for time_range in self.preferences.preferred_times):
                    reward += 0.5
                    
            return reward
            
        except Exception as e:
            self.logger.error(f"보상 계산 중 오류 발생: {str(e)}")
            return 0.0

    def train(self, num_episodes: Optional[int] = None) -> None:
        """
        SARSA 알고리즘으로 시간표 추천 모델을 학습합니다.
        
        Args:
            num_episodes (Optional[int]): 학습할 에피소드 수
        """
        try:
            num_episodes = num_episodes or self.training_config['num_episodes']
            
            for episode in range(num_episodes):
                # 초기 상태 설정
                schedule: List[Course] = []
                current_state = self._get_state(schedule)
                available_actions = self._get_available_actions(current_state)
                current_action = self._select_action(current_state, available_actions)
                
                if current_action is None:
                    continue
                    
                total_reward = 0.0
                
                # 에피소드 진행
                for step in range(self.training_config['max_steps_per_episode']):
                    # 행동 실행
                    schedule.append(current_action.course)
                    next_state = self._get_state(schedule)
                    
                    # 보상 계산
                    reward = self._calculate_reward(current_state, current_action)
                    total_reward += reward
                    
                    # 다음 행동 선택
                    next_available_actions = self._get_available_actions(next_state)
                    next_action = self._select_action(next_state, next_available_actions)
                    
                    if next_action is None:
                        break
                        
                    # Q-value 업데이트
                    current_state_key = self._get_state_key(current_state)
                    current_action_key = self._get_action_key(current_action)
                    next_state_key = self._get_state_key(next_state)
                    next_action_key = self._get_action_key(next_action)
                    
                    current_q = self.q_table[current_state_key][current_action_key]
                    next_q = self.q_table[next_state_key][next_action_key]
                    
                    self.q_table[current_state_key][current_action_key] = current_q + \
                        self.learning_rate * (reward + 
                        self.discount_factor * next_q - current_q)
                    
                    # 상태와 행동 업데이트
                    current_state = next_state
                    current_action = next_action
                    
                    # 종료 조건 체크
                    if (len(schedule) >= 8 or
                        current_state.total_credits >= self.preferences.max_credits):
                        break
                
                # ε 감소
                self.epsilon = max(
                    self.training_config['min_epsilon'],
                    self.epsilon * self.training_config['epsilon_decay']
                )
                
                if episode % 100 == 0:
                    self.logger.info(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {self.epsilon:.3f}")
                    
        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {str(e)}")

    def recommend_schedule(self) -> List[Course]:
        """
        학습된 SARSA 모델을 사용하여 시간표를 추천합니다.
        
        Returns:
            List[Course]: 추천된 시간표
        """
        try:
            schedule: List[Course] = []
            current_state = self._get_state(schedule)
            
            while True:
                # 가능한 행동들 가져오기
                available_actions = self._get_available_actions(current_state)
                
                # 최선의 행동 선택 (ε = 0)
                self.epsilon = 0.0
                best_action = self._select_action(current_state, available_actions)
                
                if best_action is None:
                    break
                    
                # 행동 실행
                schedule.append(best_action.course)
                
                # 상태 업데이트
                current_state = self._get_state(schedule)
                
                # 종료 조건 체크
                if (len(schedule) >= 8 or
                    current_state.total_credits >= self.preferences.max_credits):
                    break
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"시간표 추천 중 오류 발생: {str(e)}")
            return []

    def save_model(self) -> None:
        """
        SARSA 모델을 저장합니다.
        """
        try:
            model_data = {
                'q_table': dict(self.q_table),
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'training_config': self.training_config
            }
            
            os.makedirs('models', exist_ok=True)
            with open('models/sarsa_model.json', 'w') as f:
                json.dump(model_data, f)
            self.logger.info("SARSA 모델이 성공적으로 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {str(e)}")
            raise

    def load_model(self) -> bool:
        """
        SARSA 모델을 로드합니다.
        
        Returns:
            bool: 모델 로드 성공 여부
        """
        try:
            model_path = 'models/sarsa_model.json'
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                    
                self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
                self.learning_rate = model_data['learning_rate']
                self.discount_factor = model_data['discount_factor']
                self.epsilon = model_data['epsilon']
                self.training_config = model_data['training_config']
                
                self.logger.info("SARSA 모델이 성공적으로 로드되었습니다.")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False

    def save_schedule(self, name: str, schedule: List[Course]) -> None:
        """
        시간표를 저장합니다.
        
        Args:
            name (str): 저장할 시간표의 이름
            schedule (List[Course]): 저장할 시간표
        """
        try:
            if not name:
                raise ValueError("시간표 이름이 비어있습니다.")
            
            self.saved_schedules[name] = schedule
            schedule_path = f"schedules/{name}.json"
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(schedule_path), exist_ok=True)
            
            with open(schedule_path, 'w', encoding='utf-8') as f:
                json.dump([course.to_dict() for course in schedule], f, ensure_ascii=False, indent=2)
            self.logger.info(f"시간표가 저장되었습니다: {name}")
            
        except Exception as e:
            self.logger.error(f"시간표 저장 실패: {str(e)}")
            raise

    def load_schedule(self, name: str) -> List[Course]:
        """
        저장된 시간표를 불러옵니다.
        
        Args:
            name (str): 불러올 시간표의 이름
            
        Returns:
            List[Course]: 불러온 시간표
            
        Raises:
            ValueError: 시간표 이름이 비어있는 경우
            FileNotFoundError: 시간표 파일을 찾을 수 없는 경우
        """
        try:
            if not name:
                raise ValueError("시간표 이름이 비어있습니다.")
            
            schedule_path = f"schedules/{name}.json"
            if not os.path.exists(schedule_path):
                raise FileNotFoundError(f"시간표를 찾을 수 없습니다: {name}")
            
            with open(schedule_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            schedule = [Course.from_dict(course_data) for course_data in data]
            self.logger.info(f"시간표를 불러왔습니다: {name}")
            return schedule
            
        except (ValueError, FileNotFoundError) as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"시간표 불러오기 실패: {str(e)}")
            raise

    def generate_html(self, schedule: List[Course], output_path: str = "schedules/timetable.html") -> None:
        """
        HTML 형식의 시간표를 생성합니다.
        
        Args:
            schedule (List[Course]): 시간표
            output_path (str): 출력 파일 경로
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            html_content = generate_html_timetable(schedule)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            self.logger.info(f"HTML 시간표가 생성되었습니다: {output_path}")
            
        except Exception as e:
            self.logger.error(f"HTML 시간표 생성 실패: {str(e)}")
            raise

    def _get_available_actions(self, state: State) -> List[Action]:
        """
        현재 상태에서 가능한 행동들을 반환합니다.
        
        Args:
            state (State): 현재 상태
            
        Returns:
            List[Action]: 가능한 행동 목록
        """
        try:
            available_actions = []
            
            # 이미 선택된 과목 제외
            selected_course_codes = {course.code for course in state.selected_courses}
            
            for course in self.courses:
                if course.code in selected_course_codes:
                    continue
                    
                # 시간 충돌 체크
                if any(self.check_time_conflict(course, selected_course) 
                      for selected_course in state.selected_courses):
                    continue
                    
                # 학점 제한 체크
                if state.total_credits + course.credits > self.preferences.max_credits:
                    continue
                    
                # 과목 수 제한 체크
                if len(state.selected_courses) >= 8:
                    continue
                    
                # 가능한 시간대에 대한 행동 생성
                for time_slot in course.time_slots:
                    action = Action(
                        course=course,
                        time_slot=(time_slot.day, time_slot.start_time, time_slot.end_time)
                    )
                    available_actions.append(action)
                    
            return available_actions
            
        except Exception as e:
            self.logger.error(f"가능한 행동 생성 중 오류 발생: {str(e)}")
            return [] 