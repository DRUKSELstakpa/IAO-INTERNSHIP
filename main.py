import sys
import time
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime,timezone,UTC
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGroupBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QTabWidget, QFormLayout, QLineEdit, QTableWidget,
    QTableWidgetItem, QStatusBar, QProgressBar, QListWidget, QTextEdit,
    QFileDialog, QInputDialog, QFrame, QHeaderView, QComboBox, QTableView,
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QDialog)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QMetaObject, Q_ARG
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont, QPolygonF, QLinearGradient,
    QRadialGradient)
import pyqtgraph as pg
from typing import Tuple, List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import logging
from threading import Thread, Event, Lock
from opcua import Client, ua

# Astropy imports for coordinate conversion
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    AltAz,
    CIRS,
    Galactic,
    FK5,
    ICRS,
    get_sun,
    solar_system_ephemeris
)
from astropy.coordinates.solar_system import get_body
from astropy import units as u
from astropy.utils.iers import conf
from multiprocessing import Process, Queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SIDEREAL_RATE = 0.004178074  # Degrees per second (sidereal rate)
DEFAULT_SLEW_RATE = 30.0  # Degrees per second
DEFAULT_ACCELERATION = 1.0  # Degrees per second squared

# --- Backend Classes ---
class ConnectionStatus(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()

class TrackingRate(Enum):
    SIDEREAL = auto()
    LUNAR = auto()
    SOLAR = auto()
    CUSTOM = auto()

class TelescopeState(Enum):
    IDLE = auto()
    SLEWING = auto()
    TRACKING = auto()
    PARKING = auto()
    HOMING = auto()
    ERROR = auto()

@dataclass
class Coordinates:
    ra_deg: float  # Right Ascension in degrees (0-360)
    dec_deg: float  # Declination in degrees (-90 to 90)
    alt_deg: float  # Altitude in degrees (0-90)
    az_deg: float  # Azimuth in degrees (0-360)

@dataclass
class EnvironmentalData:
    temperature: float  # Celsius
    humidity: float  # Percentage
    pressure: float  # hPa
    wind_speed: float  # km/h
    wind_direction: float  # degrees
    dew_point: float  # Celsius
    sky_quality: float  # mag/arcsec^2

@dataclass
class CatalogObject:
    name: str
    ra_deg: float
    dec_deg: float
    obj_type: str
    magnitude: float
    size: Optional[Tuple[float, float]] = None  # Optional size in arcminutes

class TelescopeBackend(ABC):
    """Abstract base class for telescope backend implementations"""
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the telescope"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the telescope"""
        pass

    @abstractmethod
    def get_current_position(self) -> Coordinates:
        """Get current telescope coordinates"""
        pass

    @abstractmethod
    def slew_to(self, ra_deg: float, dec_deg: float) -> bool:
        """Slew telescope to specified coordinates"""
        pass

    @abstractmethod
    def start_tracking(self, rate: TrackingRate) -> bool:
        """Start tracking at specified rate"""
        pass

    @abstractmethod
    def stop_tracking(self) -> bool:
        """Stop tracking"""
        pass

    @abstractmethod
    def park(self) -> bool:
        """Park the telescope"""
        pass

    @abstractmethod
    def home(self) -> bool:
        """Home the telescope"""
        pass

    @abstractmethod
    def get_environmental_data(self) -> EnvironmentalData:
        """Get environmental sensor data"""
        pass

    @abstractmethod
    def stop_motion(self) -> bool:
        """Stop all telescope motion"""
        pass

class OpcUaTelescopeBackend(TelescopeBackend):
    """Telescope backend implementation using OPC UA with Astropy coordinate conversion"""
    def __init__(self, opc_url="opc.tcp://192.168.10.50:4840"):
        self.opc_url = opc_url
        self.client = None
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.telescope_state = TelescopeState.IDLE
        self.current_position = Coordinates(180.0, 45.0, 45.0, 180.0)
        self.target_position = Coordinates(180.0, 45.0, 45.0, 180.0)
        self.tracking_rate = TrackingRate.SIDEREAL
        self.is_tracking = False
        self.is_guiding = False
        self.stop_event = Event()
        self.tracking_thread = None
        self.slew_thread = None
        self.position_lock = Lock()
        self.tracking = False

        # Configure IERS for automatic download of Earth rotation parameters
        conf.auto_download = True

        # Initialize observatory location (default to Indian Astronomical Observatory)
        self.observatory_location = EarthLocation(
            lat=32.7908 * u.deg,
            lon=79.0002 * u.deg,
            height=4507 * u.m
        )
        self.current_time = Time.now()

        # Solar system ephemeris setup (for planetary positions)
        solar_system_ephemeris.set('jpl')

        # Initialize catalog with some common objects
        self.catalog = [
            CatalogObject("Vega", 279.2346, 38.7836, "Star", 0.03),
            CatalogObject("Altair", 297.6958, 8.8683, "Star", 0.77),
            CatalogObject("Deneb", 310.3580, 45.2803, "Star", 1.25),
            CatalogObject("M31", 10.6847, 41.2690, "Galaxy", 3.44),
            CatalogObject("M42", 83.8221, -5.3911, "Nebula", 4.0),
            CatalogObject("M45", 56.7500, 24.1167, "Cluster", 1.6),
            CatalogObject("Jupiter", 0.0, 0.0, "Planet", -2.2),
            CatalogObject("Saturn", 0.0, 0.0, "Planet", 0.5),
            CatalogObject("Mars", 0.0, 0.0, "Planet", -1.0),
            CatalogObject("Moon", 0.0, 0.0, "Moon", -12.7)
        ]

    def update_time(self, time: Optional[Time] = None) -> None:
        """Update the current time of the system"""
        with self.position_lock:
            self.current_time = Time.now() if time is None else time

    def convert_coordinates(
        self,
        coord: SkyCoord,
        to_frame: str = 'altaz',
        obstime: Optional[Time] = None
    ) -> SkyCoord:
        """Convert coordinates between different reference frames."""
        obstime = self.current_time if obstime is None else obstime

        frame_map = {
            'icrs': ICRS,
            'fk5': FK5,
            'galactic': Galactic,
            'altaz': AltAz,
            'cirs': CIRS
        }

        if to_frame.lower() not in frame_map:
            raise ValueError(f"Unsupported frame: {to_frame}")

        if to_frame.lower() == 'altaz':
            frame = AltAz(obstime=obstime, location=self.observatory_location)
        elif to_frame.lower() == 'cirs':
            frame = CIRS(obstime=obstime)
        else:
            frame = frame_map[to_frame.lower()]()

        return coord.transform_to(frame)

    def get_altaz(
        self,
        coord: SkyCoord,
        obstime: Optional[Time] = None
    ) -> Tuple[float, float]:
        """Get Altitude and Azimuth for given coordinates."""
        altaz = self.convert_coordinates(coord, 'altaz', obstime)
        return altaz.alt.degree, altaz.az.degree

    def get_tracking_coordinates(
        self,
        target: SkyCoord,
        tracking_time: Time,
        duration: u.Quantity = 1 * u.hour,
        step: u.Quantity = 1 * u.minute
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate tracking coordinates for a target over a time period."""
        times = tracking_time + np.arange(0, duration.to(u.s).value, step.to(u.s).value) * u.s
        alts, azs = [], []
        for time in times:
            alt, az = self.get_altaz(target, time)
            alts.append(alt)
            azs.append(az)

        return np.array(alts), np.array(azs)

    def get_sun_position(self, time: Optional[Time] = None) -> SkyCoord:
        """Get Sun's position at given time"""
        time = self.current_time if time is None else time
        return get_sun(time)

    def get_moon_position(self, time: Optional[Time] = None) -> SkyCoord:
        """Get Moon's position at given time"""
        time = self.current_time if time is None else time
        return get_body('moon', time, self.observatory_location)

    def is_target_visible(
        self,
        target: SkyCoord,
        min_altitude: float = 20.0,
        time: Optional[Time] = None
    ) -> bool:
        """Check if target is visible (above minimum altitude)."""
        alt, _ = self.get_altaz(target, time)
        return alt >= min_altitude

    def calculate_airmass(
        self,
        target: SkyCoord,
        time: Optional[Time] = None
    ) -> float:
        """Calculate airmass for a target."""
        alt, _ = self.get_altaz(target, time)
        if alt <= 0:
            return float('inf')
        # Simple approximation for airmass
        return 1 / np.cos(np.radians(90 - alt))

    def connect(self) -> bool:
        """Connect to the OPC UA server"""
        if self.connection_status == ConnectionStatus.CONNECTED:
            return True

        self.connection_status = ConnectionStatus.CONNECTING
        logger.info("Connecting to OPC UA server...")

        try:
            self.client = Client(self.opc_url)
            self.client.connect()
            self.connection_status = ConnectionStatus.CONNECTED
            self.telescope_state = TelescopeState.IDLE
            logger.info("Connected to OPC UA server successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to OPC UA server: {e}")
            self.connection_status = ConnectionStatus.DISCONNECTED
            return False

    def disconnect(self) -> bool:
        """Disconnect from the OPC UA server"""
        if self.connection_status == ConnectionStatus.DISCONNECTED:
            return True

        self.stop_event.set()

        # Stop any active operations
        if self.is_tracking:
            self.stop_tracking()

        if self.telescope_state == TelescopeState.SLEWING and self.slew_thread:
            self.slew_thread.join()

        try:
            if self.client:
                self.client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from OPC UA server: {e}")

        self.connection_status = ConnectionStatus.DISCONNECTED
        self.telescope_state = TelescopeState.IDLE
        logger.info("Disconnected from OPC UA server")
        return True

    def read_node(self, node_str):
        """Read a value from an OPC UA node"""
        try:
            node = self.client.get_node(node_str)
            return node.get_value()
        except Exception as e:
            logger.error(f"Error reading node {node_str}: {e}")
            return None

    def write_boolean(self, node_str, value):
        """Write a boolean value to an OPC UA node"""
        try:
            node = self.client.get_node(f"ns=4;s={node_str}")
            var = ua.Variant(value, ua.VariantType.Boolean)
            node.set_value(var)
            return True
        except Exception as e:
            logger.error(f"Error writing boolean to {node_str}: {e}")
            return False

    def write_double(self, node_str, value):
        """Write a double value to an OPC UA node"""
        try:
            node = self.client.get_node(f"ns=4;s={node_str}")
            var = ua.Variant(value, ua.VariantType.Double)
            node.set_value(var)
            return True
        except Exception as e:
            logger.error(f"Error writing double to {node_str}: {e}")
            return False

    def get_current_position(self) -> Coordinates:
        """Get current telescope coordinates from OPC UA server"""
        if self.connection_status != ConnectionStatus.CONNECTED:
            return self.current_position

        # Read actual positions from OPC UA
        az_actpos = self.read_node("ns=4;s=|var|PAC320-CME21-3A.Application.PLC_PRG.Az_actpos")
        el_actpos = self.read_node("ns=4;s=|var|PAC320-CME21-3A.Application.PLC_PRG.EL_actpos")

        with self.position_lock:
            if az_actpos is not None:
                self.current_position.az_deg = az_actpos
            if el_actpos is not None:
                self.current_position.alt_deg = el_actpos

            # For simulation purposes, we'll calculate RA/DEC from ALT/AZ
            self._update_radec()
            return self.current_position

    def _update_radec(self):
        """Calculate RA/DEC from ALT/AZ using Astropy"""
        # Create AltAz frame
        altaz = AltAz(
            alt=self.current_position.alt_deg * u.deg,
            az=self.current_position.az_deg * u.deg,
            obstime=self.current_time,
            location=self.observatory_location
        )

        # Convert to ICRS (RA/DEC)
        skycoord = SkyCoord(altaz.transform_to(ICRS()))

        self.current_position.ra_deg = skycoord.ra.degree
        self.current_position.dec_deg = skycoord.dec.degree

    def slew_to(self, ra_deg: float, dec_deg: float) -> bool:
        """Slew telescope to specified coordinates"""
        if self.connection_status != ConnectionStatus.CONNECTED:
            logger.error("Cannot slew - telescope not connected")
            return False

        if self.telescope_state == TelescopeState.SLEWING:
            logger.warning("Already slewing - stopping current slew")
            self.stop_event.set()
            if self.slew_thread:
                self.slew_thread.join()

        with self.position_lock:
            self.target_position.ra_deg = ra_deg % 360
            self.target_position.dec_deg = max(-90, min(90, dec_deg))

        # Start slew in a separate thread
        self.stop_event.clear()
        self.slew_thread = Thread(target=self._slew_to_target)
        self.slew_thread.start()
        return True

    def _slew_to_target(self):
        """Thread function for slewing to target using OPC UA"""
        self.telescope_state = TelescopeState.SLEWING
        logger.info(f"Starting slew to RA: {self.target_position.ra_deg:.2f}, DEC: {self.target_position.dec_deg:.2f}")

        try:
            # Calculate target ALT/AZ
            self._calculate_target_altaz()

            # Movement command sequence
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.Az_halt2", False)
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.EL_halt", False)
            self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.Az_Acc", DEFAULT_ACCELERATION)
            self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.Az_velocity", DEFAULT_SLEW_RATE)
            self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.EL_Acc", DEFAULT_ACCELERATION)
            self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.EL_velocity", DEFAULT_SLEW_RATE)
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.Az_moveAbsolute", False)
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.EL_moveAbsolute", False)
            self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.Az_MoveAbsolutePosition", self.target_position.az_deg)
            self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.EL_MoveAbsolutePosition", self.target_position.alt_deg)
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.Az_moveAbsolute", True)
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.EL_moveAbsolute", True)

            # Wait for slew to complete
            while not self.stop_event.is_set():
                az_done = self.read_node("ns=4;s=|var|PAC320-CWE21-3A.Application.PLC_PRG.MC_MoveAbsolute_Az_Done")
                el_done = self.read_node("ns=4;s=|var|PAC320-CWE21-3A.Application.PLC_PRG.MC_MoveAbsolute_EL_Done")
                if az_done and el_done:
                    break
                time.sleep(0.1)

            if not self.stop_event.is_set():
                with self.position_lock:
                    self.current_position.ra_deg = self.target_position.ra_deg
                    self.current_position.dec_deg = self.target_position.dec_deg
                    self.current_position.az_deg = self.target_position.az_deg
                    self.current_position.alt_deg = self.target_position.alt_deg
                logger.info("Slew completed successfully")
            else:
                logger.info("Slew interrupted by user")

        except Exception as e:
            logger.error(f"Error during slew: {e}")

        self.telescope_state = TelescopeState.IDLE

    def _calculate_target_altaz(self):
        """Calculate target ALT/AZ from RA/DEC using Astropy"""
        # Create SkyCoord for target
        target_coord = SkyCoord(
            ra=self.target_position.ra_deg * u.deg,
            dec=self.target_position.dec_deg * u.deg,
            frame='icrs'
        )

        # Convert to AltAz
        altaz = target_coord.transform_to(AltAz(
            obstime=self.current_time,
            location=self.observatory_location
        ))

        with self.position_lock:
            self.target_position.alt_deg = altaz.alt.degree
            self.target_position.az_deg = altaz.az.degree

    def start_tracking(self, rate: TrackingRate) -> bool:
        """Start tracking using multiprocessing"""
        if self.connection_status != ConnectionStatus.CONNECTED:
            logger.error("Cannot start tracking - telescope not connected")
            return False

        if self.is_tracking:
            self.stop_tracking()

        self.tracking_rate = rate
        self.is_tracking = True

        # Create multiprocessing queue and process
        self.tracking_queue = Queue()
        self.tracking_process = Process(
            target=tracking_process,  # this must be defined outside the class
            args=(
                self.tracking_queue,
                self.target_position.ra_deg,
                self.tracking_rate.name,
                {
                    'lat': self.observatory_location.lat.degree,
                    'lon': self.observatory_location.lon.degree,
                    'height': self.observatory_location.height.value
                },
                time.time()
            )
        )
        self.tracking_process.start()

        # Start listener thread to receive updates
        self.tracking_listener = Thread(target=self._process_tracking_updates)
        self.tracking_listener.start()

        logger.info(f"Started tracking using multiprocessing at {rate.name} rate")
        return True

    def _process_tracking_updates(self):
        location = EarthLocation(
            lat=self.observatory_location.lat.degree * u.deg,
            lon=self.observatory_location.lon.degree * u.deg,
            height=self.observatory_location.height.value * u.m
        )

        start_time = time.time()

        while self.is_tracking:
            try:
                now = Time.now()
                delta_time = time.time() - start_time

                # Adjust RA based on tracking mode
                ra_deg = self.target_position.ra_deg
                dec_deg = self.target_position.dec_deg

                if self.tracking_rate == "SIDEREAL":
                    ra_deg = (ra_deg + SIDEREAL_RATE * delta_time) % 360
                elif self.tracking_rate == "LUNAR":
                    ra_deg = (ra_deg + SIDEREAL_RATE * 0.966 * delta_time) % 360
                elif self.tracking_rate == "SOLAR":
                    ra_deg = (ra_deg + SIDEREAL_RATE * 1.0027 * delta_time) % 360

                # Perform RA/DEC ➤ ALT/AZ conversion
                sky_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                altaz = sky_coord.transform_to(AltAz(obstime=now, location=location))

                az = altaz.az.degree
                alt = altaz.alt.degree

                self.target_position.ra_deg = ra_deg
                self.target_position.alt_deg = alt
                self.target_position.az_deg = az

                # Stop current motion and set parameters
                self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.Az_halt2", False)
                self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.EL_halt", False)

                self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.Az_Acc", DEFAULT_ACCELERATION)
                self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.Az_velocity", DEFAULT_SLEW_RATE)
                self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.EL_Acc", DEFAULT_ACCELERATION)
                self.write_double("|var|PAC320-CWE21-3A.Application.global_variable.EL_velocity", DEFAULT_SLEW_RATE)

                self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.Az_moveAbsolute", False)
                self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.EL_moveAbsolute", False)

                self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.Az_MoveAbsolutePosition", az)
                self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.EL_MoveAbsolutePosition", alt)

                self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.Az_moveAbsolute", True)
                self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.EL_moveAbsolute", True)

                logger.info(f"[Tracking] RA: {ra_deg:.3f}, DEC: {dec_deg:.3f} → ALT: {alt:.2f}°, AZ: {az:.2f}°")

            except Exception as e:
                logger.error(f"Tracking update error: {e}")

            time.sleep(0.5)

    def stop_tracking(self) -> bool:
        if not self.is_tracking:
            return False

        self.is_tracking = False

        if hasattr(self, 'tracking_queue') and self.tracking_queue:
            self.tracking_queue.put("STOP")

        if hasattr(self, 'tracking_process') and self.tracking_process:
            self.tracking_process.join()

        if hasattr(self, 'tracking_listener') and self.tracking_listener:
            self.tracking_listener.join()

        logger.info("Tracking stopped (multiprocessing)")
        return True


    def _tracking_loop(self):
        """Advanced tracking control loop with continuous coordinate updates"""
        last_update_time = time.time()

        while self.is_tracking and not self.stop_event.is_set():
            try:
                current_time = time.time()
                dt = current_time - last_update_time
                last_update_time = current_time

                # Update system time
                self.update_time()

                with self.position_lock:
                    # For sidereal tracking, we need to adjust RA continuously
                    if self.tracking_rate == TrackingRate.SIDEREAL:
                        # Calculate the apparent motion of the sky
                        delta_ra = SIDEREAL_RATE * dt  # degrees per second (sidereal rate)
                        # Update target RA
                        self.target_position.ra_deg = (self.target_position.ra_deg + delta_ra) % 360
                        # Recalculate ALT/AZ
                        self._calculate_target_altaz()

                        # Send updated position to mount
                        self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.Az_MoveAbsolutePosition", self.target_position.az_deg)
                        self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.EL_MoveAbsolutePosition", self.target_position.alt_deg)

                    # For other tracking rates, implement similar logic with appropriate rates
                    elif self.tracking_rate == TrackingRate.LUNAR:
                        # Lunar tracking rate is slightly slower than sidereal
                        delta_ra = SIDEREAL_RATE * 0.966 * dt
                        self.target_position.ra_deg = (self.target_position.ra_deg + delta_ra) % 360
                        self._calculate_target_altaz()
                        self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.Az_MoveAbsolutePosition", self.target_position.az_deg)
                        self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.EL_MoveAbsolutePosition", self.target_position.alt_deg)

                    elif self.tracking_rate == TrackingRate.SOLAR:
                        # Solar tracking rate is slightly faster than sidereal
                        delta_ra = SIDEREAL_RATE * 1.0027 * dt
                        self.target_position.ra_deg = (self.target_position.ra_deg + delta_ra) % 360
                        self._calculate_target_altaz()
                        self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.Az_MoveAbsolutePosition", self.target_position.az_deg)
                        self.write_double("|var|PAC320-CWE21-3A.Application.PLC_PRG.EL_MoveAbsolutePosition", self.target_position.alt_deg)

                # Adjust sleep time based on tracking rate
                sleep_time = max(0.05, 1.0 - (time.time() - current_time))
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                time.sleep(1)  # Prevent tight loop on errors

    def park(self) -> bool:
        """Park the telescope"""
        if self.connection_status != ConnectionStatus.CONNECTED:
            logger.error("Cannot park - telescope not connected")
            return False

        logger.info("Parking telescope...")

        try:
            # NodeId for Az_starthome
            nodeIdAz = self.client.get_node("ns=4;s=|var|PAC320-CWE21-3A.Application.global_variable.Az_starthome")
            # Write True to the node
            nodeIdAz.set_value(True, ua.VariantType.Boolean)

            # NodeId for EL_starthome
            nodeIdEl = self.client.get_node("ns=4;s=|var|PAC320-CWE21-3A.Application.global_variable.EL_starthome")
            # Write True to the node
            nodeIdEl.set_value(True, ua.VariantType.Boolean)

            logger.info("Parking command sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send park command: {e}")
            return False

    def home(self) -> bool:
        """Home the telescope"""
        if self.connection_status != ConnectionStatus.CONNECTED:
            logger.error("Cannot home - telescope not connected")
            return False

        logger.info("Homing telescope...")

        try:
            # NodeId for Az_home
            nodeIdAz = self.client.get_node("ns=4;s=|var|PAC320-CWE21-3A.Application.global_variable.Az_home")
            # Write True to the node
            nodeIdAz.set_value(True, ua.VariantType.Boolean)
            # NodeId for EL_home
            nodeIdEl = self.client.get_node("ns=4;s=|var|PAC320-CWE21-3A.Application.global_variable.EL_home")
            # Write True to the node
            nodeIdEl.set_value(True, ua.VariantType.Boolean)
            logger.info("Homing command sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send home command: {e}")
            return False

    def get_environmental_data(self) -> EnvironmentalData:
        """Get environmental sensor data (simulated)"""
        t = time.time() / 3600  # Hours since epoch

        # Simulate realistic variations
        temp = 15 + 5 * math.sin(t) + random.uniform(-0.5, 0.5)
        humidity = 45 + 10 * math.sin(t/2) + random.uniform(-2, 2)
        pressure = 1013 + random.uniform(-2, 2)
        wind = 5.2 + random.uniform(-1, 1)
        wind_dir = random.uniform(0, 360)

        # Calculate dew point
        dew_point = temp - (100 - humidity) / 5

        # Simulate sky quality (lower is better)
        sky_quality = 20.5 + 0.5 * math.sin(t/3) + random.uniform(-0.2, 0.2)

        return EnvironmentalData(
            temperature=temp,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind,
            wind_direction=wind_dir,
            dew_point=dew_point,
            sky_quality=sky_quality
        )

    def stop_motion(self) -> bool:
        """Stop all telescope motion"""
        if self.connection_status != ConnectionStatus.CONNECTED:
            return False

        try:
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.Az_halt2", True)
            self.write_boolean("|var|PAC320-CWE21-3A.Application.global_variable.EL_halt", True)
            logger.info("Motion stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop motion: {e}")
            return False

    def search_catalog(self, search_term: str) -> List[CatalogObject]:
        """Search the catalog for objects matching the term"""
        return [obj for obj in self.catalog if search_term.lower() in obj.name.lower()]

class StarInfoDialog(QDialog):
    def __init__(self, star, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Star Information")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Name: {star['name']}"))
        layout.addWidget(QLabel(f"RA: {np.degrees(star['ra']):.2f}°"))
        layout.addWidget(QLabel(f"Dec: {np.degrees(star['dec']):.2f}°"))
        layout.addWidget(QLabel(f"Magnitude: {star['vmag']:.2f}"))
        layout.addWidget(QLabel(f"Spectral Type: {star['spType']}"))

# New SearchTab class
class SearchTab(QWidget):
    def __init__(self, sky_map, parent=None):
        super().__init__(parent)
        self.sky_map = sky_map  # Reference to SkyMapTab instance
        layout = QVBoxLayout(self)

        # Search input and button
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter star name (e.g., Vega)")
        search_layout.addWidget(self.search_input)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_stars)
        search_layout.addWidget(self.search_button)

        layout.addLayout(search_layout)

        # Search results
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_clicked)
        layout.addWidget(self.results_list)

    def search_stars(self):
        """Search for stars by name and display results."""
        search_text = self.search_input.text().strip().lower()
        self.results_list.clear()
        if not search_text:
            return

        matching_stars = [
            star for star in self.sky_map.stars
            if search_text in star['name'].lower()
        ]

        for star in matching_stars:
            item = QListWidgetItem(f"{star['name']} (Mag: {star['vmag']:.2f})")
            item.setData(Qt.UserRole, star)  # Store star dict in item
            self.results_list.addItem(item)

        if not matching_stars:
            self.results_list.addItem("No stars found")

    def on_result_clicked(self, item):
        """Highlight the selected star on the sky map."""
        star = item.data(Qt.UserRole)
        if star:
            self.sky_map.highlight_star(star)

# Updated SkyMapTab class
class SkyMapTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-200, -200, 400, 400)
        self.view = QGraphicsView(self.scene)
        self.view.setMouseTracking(True)
        layout.addWidget(self.view)
        self.info_text = None
        self.stars = self.load_star_catalog()
        self.star_positions = {}
        self.current_position_item = None
        self.target_position_item = None
        self.highlighted_star_item = None  # New item for highlighted star
        print("Calling plot_stars() in __init__")
        self.plot_stars()
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_star_positions)
        self.update_timer.start(60000)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def load_star_catalog(self):
        try:
            df = pd.read_csv('starcatalog.csv')
            stars = []
            for _, row in df.iterrows():
                ra = (float(row.iloc[19]) + float(row.iloc[20])/60 + float(row.iloc[21])/3600) * np.pi / 12
                dec_sign = -1 if str(row.iloc[22]).strip() == '-' else 1
                dec = ((float(row.iloc[23]) + float(row.iloc[24])/60 + float(row.iloc[25])/3600)) * dec_sign * np.pi / 180
                vmag = float(row.iloc[28])
                spectral_type = str(row.iloc[37]).strip()
                sp_type = spectral_type[0] if spectral_type else 'U'
                if vmag <= 6:
                    stars.append({
                        'hr': int(row.iloc[0]),
                        'name': str(row.iloc[1]).strip(),
                        'ra': ra,
                        'dec': dec,
                        'vmag': vmag,
                        'spType': sp_type
                    })
            print(f'loaded stars: {len(stars)}')
            return stars
        except Exception as e:
            print(f"Error loading star catalog: {e}")
            return [
                {'hr': 1, 'name': 'Vega', 'ra': np.radians(279.2346), 'dec': np.radians(38.7836), 'vmag': 0.03, 'spType': 'A'},
                {'hr': 2, 'name': 'Altair', 'ra': np.radians(297.6958), 'dec': np.radians(8.8683), 'vmag': 0.77, 'spType': 'A'},
                {'hr': 3, 'name': 'Deneb', 'ra': np.radians(310.3580), 'dec': np.radians(45.2803), 'vmag': 1.25, 'spType': 'A'}
            ]

    def get_star_color(self, sp_type):
        colors = {
            'O': QColor(156, 178, 255),
            'B': QColor(170, 191, 255),
            'A': QColor(202, 215, 255),
            'F': QColor(248, 247, 255),
            'G': QColor(255, 244, 234),
            'K': QColor(255, 210, 161),
            'M': QColor(255, 204, 111),
            'U': QColor(255, 255, 255)
        }
        return colors.get(sp_type, QColor(255, 255, 255))

    def get_alt_az(self, ra, dec):
        star_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame='icrs')
        location = EarthLocation(lat=32.7908 * u.deg, lon=79.0002 * u.deg, height=4507 * u.m)
        current_time = Time(datetime.now(UTC))
        altaz_frame = AltAz(obstime=current_time, location=location)
        star_altaz = star_coord.transform_to(altaz_frame)
        return star_altaz.alt.rad, star_altaz.az.rad

    def plot_stars(self):
        print("plot_stars called")
        self.scene.clear()
        self.star_positions = {}
        self.current_position_item = None
        self.target_position_item = None
        self.highlighted_star_item = None  # Reset highlighted star
        for alt in range(0, 91, 15):
            radius = (90 - alt) * 2
            self.scene.addEllipse(-radius, -radius, radius*2, radius*2,
                                 QPen(QColor(100, 100, 100)))
        radius = 180
        font_size = 12
        marker_color = QColor(200, 200, 200)
        north_text = self.scene.addText("N")
        north_text.setDefaultTextColor(marker_color)
        north_text.setFont(QFont("Arial", font_size))
        north_text.setPos(-font_size/2, -radius-font_size*2)
        south_text = self.scene.addText("S")
        south_text.setDefaultTextColor(marker_color)
        south_text.setFont(QFont("Arial", font_size))
        south_text.setPos(-font_size/2, radius+font_size/2)
        east_text = self.scene.addText("E")
        east_text.setDefaultTextColor(marker_color)
        east_text.setFont(QFont("Arial", font_size))
        east_text.setPos(radius+font_size/2, -font_size/2)
        west_text = self.scene.addText("W")
        west_text.setDefaultTextColor(marker_color)
        west_text.setFont(QFont("Arial", font_size))
        west_text.setPos(-radius-font_size*1.5, -font_size/2)
        count = 0
        for idx, star in enumerate(self.stars):
            alt_rad, az = self.get_alt_az(star['ra'], star['dec'])
            alt_deg = np.degrees(alt_rad)
            if alt_deg > 0:
                r = (90 - alt_deg) * 2
                theta = az + np.pi
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                self.star_positions[idx] = (x, y)
                min_size = 0.5
                max_size = 2.5
                scale = (star['vmag'] - 0) / 5
                size = min_size + (max_size - min_size) * np.exp(-2.5 * scale)
                color = self.get_star_color(star['spType'])
                star_item = QGraphicsEllipseItem(x-size/2, y-size/2, size, size)
                star_item.setBrush(QBrush(color))
                star_item.setPen(QPen(color))
                self.scene.addItem(star_item)
                count += 1
        print(f"Stars plotted: {count}")
        if hasattr(self, 'current_position'):
            self.set_current_position(self.current_position[0], self.current_position[1])
        if hasattr(self, 'target_position'):
            self.set_target_position(self.target_position[0], self.target_position[1])

    def set_current_position(self, ra_deg, dec_deg):
        self.current_position = (ra_deg, dec_deg)
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        alt_rad, az_rad = self.get_alt_az(ra_rad, dec_rad)
        alt_deg = np.degrees(alt_rad)
        if alt_deg > 0:
            r = (90 - alt_deg) * 2
            theta = az_rad + np.pi
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            if self.current_position_item:
                self.scene.removeItem(self.current_position_item)
            size = 10
            pen = QPen(QColor(255, 0, 0), 2)
            group = QGraphicsItemGroup()
            group.addToGroup(self.scene.addLine(x - size, y - size, x + size, y + size, pen))
            group.addToGroup(self.scene.addLine(x - size, y + size, x + size, y - size, pen))
            self.scene.addItem(group)
            self.current_position_item = group

    def set_target_position(self, ra_deg, dec_deg):
        self.target_position = (ra_deg, dec_deg)
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        alt_rad, az_rad = self.get_alt_az(ra_rad, dec_rad)
        alt_deg = np.degrees(alt_rad)
        if alt_deg > 0:
            r = (90 - alt_deg) * 2
            theta = az_rad + np.pi
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            if self.target_position_item:
                self.scene.removeItem(self.target_position_item)
            size = 8
            pen = QPen(QColor(0, 255, 0), 2)
            self.target_position_item = self.scene.addEllipse(
                x - size / 2, y - size / 2, size, size, pen
            )

    def highlight_star(self, star):
        """Highlight a star on the sky map by drawing a distinct marker."""
        alt_rad, az_rad = self.get_alt_az(star['ra'], star['dec'])
        alt_deg = np.degrees(alt_rad)
        if alt_deg > 0:
            r = (90 - alt_deg) * 2
            theta = az_rad + np.pi
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            if self.highlighted_star_item:
                self.scene.removeItem(self.highlighted_star_item)
            size = 10  # Larger size for visibility
            pen = QPen(QColor(255, 255, 0), 2)  # Yellow outline
            brush = QBrush(QColor(255, 255, 0, 100))  # Semi-transparent yellow fill
            self.highlighted_star_item = self.scene.addEllipse(
                x - size / 2, y - size / 2, size, size, pen, brush
            )
            # Center the view on the star
            self.view.centerOn(x, y)

    def update_star_positions(self):
        print("updating stars")
        self.plot_stars()

    def mousePressEvent(self, event):
        scene_pos = self.view.mapToScene(event.pos())
        closest_star = self.find_closest_star(scene_pos.x(), scene_pos.y())
        if closest_star:
            dialog = StarInfoDialog(closest_star, self)
            dialog.exec()

    def find_closest_star(self, x, y):
        min_distance = float('inf')
        closest_star = None
        max_click_distance = 10
        for idx, star in enumerate(self.stars):
            pos = self.star_positions.get(idx)
            if pos is None:
                continue
            star_x, star_y = pos
            distance = np.sqrt((x - star_x)**2 + (y - star_y)**2)
            if distance < min_distance and distance < max_click_distance:
                min_distance = distance
                closest_star = star
        return closest_star

class TelescopeControlSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STAWA - Telescope Control System")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize backend
        self.backend = OpcUaTelescopeBackend("opc.tcp://192.168.10.50:4840")

        # Initialize system state
        self.connected = False
        self.tracking = False
        self.guiding = False
        self.slewing = False
        self.capturing = False
        self.live_view_active = False
        self.tracking_rate = "Sidereal"
        self.park_position = "Home"

        # Configure UI
        self.setup_ui()
        self.setup_timers()

    def setup_ui(self):
        """Initialize all UI components"""
        self.setup_styles()
        self.setWindowTitle("Telescope Control System")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_main_window()
        self.setup_status_bar()

    def setup_styles(self):
        """Configure the application style"""
        self.setStyleSheet("""
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0d1117, stop:0.5 #161b22, stop:1 #21262d);
        }
        QWidget {
            background: transparent;
            color: #e6edf3;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #238636, stop:1 #1f72a2);
            border: 2px solid #2ea043;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            color: white;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2ea043, stop:1 #238636);
        }
        QPushButton:pressed {
            background: #1172a2;
        }
        QPushButton:disabled {
            background: #6e7681;
            border-color: #6e7681;
            color: #8d96a0;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox {
            background: #21262d;
            border: 2px solid #30363d;
            border-radius: 6px;
            padding: 8px;
            color: #e6edf3;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border-color: #58afff;
        }
        QComboBox {
            background: #21262d;
            border: 2px solid #30363d;
            border-radius: 6px;
            padding: 8px;
            color: #e6edf3;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #e6edf3;
        }
        QGroupBox {
            border: 2px solid #30363d;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            color: #58a6ff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px 0 8px;
        }
        QTabWidget::pane {
            border: 2px solid #30363d;
            border-radius: 8px;
        }
        QTabBar::tab {
            background: #21262d;
            border: 2px solid #30363d;
            border-bottom: none;
            border-radius: 6px 6px 0 0;
            padding: 8px 16px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #58a6ff;
            color: #0d1117;
            font-weight: bold;
        }
        QProgressBar {
            border: 2px solid #30363d;
            border-radius: 6px;
            text-align: center;
            background: #21262d;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #f85149, stop:0.5 #da3633, stop:1 #b62324);
            border-radius: 4px;
        }
        QSlider::groove:horizontal {
            border: 2px solid #30363d;
            height: 8px;
            background: #21262d;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #58a6ff;
            border: 2px solid #116feb;
            width: 18px;
            border-radius: 9px;
            margin: -5px 0;
        }
        QSlider::handle:horizontal:hover {
            background: #79c0ff;
        }
        QTableWidget {
            background: #21262d;
            border: 2px solid #30363d;
            border-radius: 6px;
            gridline-color: #30363d;
        }
        QTableWidget::item {
            padding: 5px;
            border-bottom: 1px solid #30363d;
        }
        QTableWidget::item:selected {
            background: #58a6ff;
            color: #0d1117;
        }
        QHeaderView::section {
            background: #161b22;
            padding: 5px;
            border: 1px solid #30363d;
        }
        QTextEdit {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 5px;
            font-family: 'Courier New', monospace;
            font-size: 10px;
        }
        """)
        # Additional style configurations
        pg.setConfigOption('background', '#0d1117')
        pg.setConfigOption('foreground', '#e6edf3')

    def setup_main_window(self):
        """Create the main window layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Create and add panels
        main_layout.addWidget(self.create_control_panel(), 1)
        main_layout.addWidget(self.create_visualization_panel(), 2)
        main_layout.addWidget(self.create_status_panel(), 1)

    def setup_status_bar(self):
        """Configure the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connection status indicator
        self.connection_indicator = QLabel()
        self.update_connection_indicator()

        # Battery indicator
        self.battery_indicator = QProgressBar()
        self.battery_indicator.setRange(0, 100)
        self.battery_indicator.setValue(85)
        self.battery_indicator.setFormat("Battery: %p%")
        self.battery_indicator.setMaximumWidth(150)

        # System time
        self.system_time = QLabel()
        self.update_system_time()

        # Add widgets to status bar
        self.status_bar.addPermanentWidget(self.connection_indicator)
        self.status_bar.addPermanentWidget(self.battery_indicator)
        self.status_bar.addPermanentWidget(self.system_time)

    def create_control_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Title
        title = QLabel("TELESCOPE CONTROL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #58a6ff;
            padding: 10px;
            background: rgba(88, 166, 255, 0.1);
            border-radius: 8px;
        """)
        layout.addWidget(title)

        # Create tab widget for all control sections
        self.control_tabs = QTabWidget()

        # Mount Tab
        mount_tab = self.create_mount_tab()
        self.control_tabs.addTab(mount_tab, "Mount")

        # Catalog Tab
        catalog_tab = self.create_catalog_tab()
        self.control_tabs.addTab(catalog_tab, "Catalog")

        # Target Coordinates Tab
        target_tab = self.create_target_tab()
        self.control_tabs.addTab(target_tab, "Target")

        # Focus Position Tab
        focus_pos_tab = self.create_focus_position_tab()
        self.control_tabs.addTab(focus_pos_tab, "Focus")

        # Rotate Position Tab
        rotate_pos_tab = self.create_rotate_position_tab()
        self.control_tabs.addTab(rotate_pos_tab, "Rotate")

        # Mount Debug Tab
        mount_debug_tab = self.create_mount_debug_tab()
        self.control_tabs.addTab(mount_debug_tab, "Debug")

        layout.addWidget(self.control_tabs)
        return panel

    def create_mount_tab(self):
        """Create the Mount control tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Connection group
        connection_group = QGroupBox("Connection")
        connection_layout = QVBoxLayout(connection_group)

        self.connect_btn = QPushButton("Connect Telescope")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_btn)

        self.connection_status = QLabel("Status: Disconnected")
        self.connection_status.setStyleSheet("color: #f85149; font-weight: bold;")
        connection_layout.addWidget(self.connection_status)

        layout.addWidget(connection_group)

        # Tracking group
        tracking_group = QGroupBox("Tracking")
        tracking_layout = QVBoxLayout(tracking_group)

        self.tracking_btn = QPushButton("Start Tracking")
        self.tracking_btn.clicked.connect(self.toggle_tracking)
        tracking_layout.addWidget(self.tracking_btn)

        self.tracking_rate_combo = QComboBox()
        self.tracking_rate_combo.addItems(["Sidereal", "Lunar", "Solar", "Custom"])
        self.tracking_rate_combo.currentTextChanged.connect(self.update_tracking_rate)
        tracking_layout.addWidget(self.tracking_rate_combo)

        layout.addWidget(tracking_group)

        # Guiding group
        guide_group = QGroupBox("Auto-Guide")
        guide_layout = QVBoxLayout(guide_group)

        self.guide_btn = QPushButton("Start Guiding")
        self.guide_btn.clicked.connect(self.toggle_guiding)
        guide_layout.addWidget(self.guide_btn)

        self.guide_status = QLabel("Guide Status: Stopped")
        guide_layout.addWidget(self.guide_status)

        layout.addWidget(guide_group)

        # Park/Home group
        park_group = QGroupBox("Park/Home")
        park_layout = QHBoxLayout(park_group)

        self.park_btn = QPushButton("Park")
        self.park_btn.clicked.connect(self.park_telescope)
        park_layout.addWidget(self.park_btn)

        self.home_btn = QPushButton("Home")
        self.home_btn.clicked.connect(self.home_telescope)
        park_layout.addWidget(self.home_btn)

        self.unpark_btn = QPushButton("Unpark")
        self.unpark_btn.clicked.connect(self.unpark_telescope)
        park_layout.addWidget(self.unpark_btn)

        layout.addWidget(park_group)

        # Coordinate input group
        coord_group = QGroupBox("Manual Coordinates")
        coord_layout = QFormLayout(coord_group)

        self.ra_input = QLineEdit()
        self.ra_input.setPlaceholderText("HH:MM:SS or degrees")
        coord_layout.addRow("Right Ascension:", self.ra_input)

        self.dec_input = QLineEdit()
        self.dec_input.setPlaceholderText("DD:MM:SS or degrees")
        coord_layout.addRow("Declination:", self.dec_input)

        self.goto_coord_btn = QPushButton("Go To Coordinates")
        self.goto_coord_btn.clicked.connect(self.goto_coordinates)
        coord_layout.addRow(self.goto_coord_btn)

        layout.addWidget(coord_group)

        # Stop button
        self.stop_btn = QPushButton("Emergency Stop")
        self.stop_btn.setStyleSheet("background-color: #f85149; border-color:#da3633;")
        self.stop_btn.clicked.connect(self.emergency_stop)
        layout.addWidget(self.stop_btn)

        layout.addStretch()
        return tab

    def create_target_tab(self):
        """Create the Target Coordinates tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Target coordinates group
        target_group = QGroupBox("Target Coordinates")
        target_layout = QFormLayout(target_group)

        self.target_ra_input = QLineEdit()
        self.target_ra_input.setPlaceholderText("HH:MM:SS or degrees")
        target_layout.addRow("Right Ascension:", self.target_ra_input)

        self.target_dec_input = QLineEdit()
        self.target_dec_input.setPlaceholderText("DD:MM:SS or degrees")
        target_layout.addRow("Declination:", self.target_dec_input)

        self.target_ra_display = QLabel("12:00:00")  # Add RA display
        target_layout.addRow("RA (Display):", self.target_ra_display)

        self.target_dec_display = QLabel("+45:00:00")  # Add DEC display
        target_layout.addRow("DEC (Display):", self.target_dec_display)

        self.target_az_display = QLabel("180.0000")
        target_layout.addRow("Azimuth:", self.target_az_display)

        self.target_alt_display = QLabel("45.0000")
        target_layout.addRow("Altitude:", self.target_alt_display)

        # Buttons
        button_layout = QHBoxLayout()
        self.get_altaz_btn = QPushButton("Get Alt/Az")
        self.get_altaz_btn.clicked.connect(self.get_altaz)
        button_layout.addWidget(self.get_altaz_btn)

        self.goto_altaz_btn = QPushButton("Go To Alt/Az")
        self.goto_altaz_btn.clicked.connect(self.goto_altaz)
        button_layout.addWidget(self.goto_altaz_btn)

        self.save_altaz_btn = QPushButton("Save")
        self.save_altaz_btn.clicked.connect(self.save_altaz)
        button_layout.addWidget(self.save_altaz_btn)

        target_layout.addRow(button_layout)

        layout.addWidget(target_group)

        # Current position group
        current_group = QGroupBox("Current Position")
        current_layout = QFormLayout(current_group)

        self.current_ra_display = QLabel("12:00:00")
        self.current_dec_display = QLabel("+45:00:00")
        self.current_az_display = QLabel("180.0000")
        self.current_alt_display = QLabel("45.0000")

        current_layout.addRow("Right Ascension:", self.current_ra_display)
        current_layout.addRow("Declination:", self.current_dec_display)
        current_layout.addRow("Azimuth:", self.current_az_display)
        current_layout.addRow("Altitude:", self.current_alt_display)

        layout.addWidget(current_group)

        # Following error graph
        self.following_error_plot = pg.PlotWidget()
        self.setup_following_error_plot()
        layout.addWidget(self.following_error_plot)

        return tab

    def create_catalog_tab(self):
        """Create the Catalog tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Search group
        search_group = QGroupBox("Search")
        search_layout = QHBoxLayout(search_group)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search catalog...")
        search_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search_catalog)
        search_layout.addWidget(self.search_btn)

        layout.addWidget(search_group)

        # Catalog table
        self.catalog_table = QTableWidget()
        self.catalog_table.setColumnCount(5)
        self.catalog_table.setHorizontalHeaderLabels(['Name', 'RA', 'DEC', 'Type', 'Magnitude'])
        self.catalog_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.catalog_table.setSelectionBehavior(QTableView.SelectRows)
        self.catalog_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Populate with sample data
        self.populate_catalog()

        layout.addWidget(self.catalog_table)

        # Catalog actions
        catalog_actions = QHBoxLayout()
        self.goto_catalog_btn = QPushButton("Go To")
        self.goto_catalog_btn.clicked.connect(self.goto_catalog_object)
        catalog_actions.addWidget(self.goto_catalog_btn)

        self.add_to_list_btn = QPushButton("Add to List")
        self.add_to_list_btn.clicked.connect(self.add_catalog_to_list)
        catalog_actions.addWidget(self.add_to_list_btn)

        layout.addLayout(catalog_actions)

        return tab

    def create_focus_position_tab(self):
        """Create the Focus Position tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Focus control group
        control_group = QGroupBox("Focus Control")
        control_layout = QVBoxLayout(control_group)

        # Focus position slider
        self.focus_slider = QSlider(Qt.Horizontal)
        self.focus_slider.setRange(0, 10000)
        self.focus_slider.setValue(5000)
        self.focus_slider.valueChanged.connect(self.set_focus_position)
        control_layout.addWidget(self.focus_slider)

        # Focus position display
        self.focus_pos_display = QLabel("5000")
        self.focus_pos_display.setAlignment(Qt.AlignCenter)
        self.focus_pos_display.setStyleSheet("font-size: 24px; font-weight: bold;")
        control_layout.addWidget(self.focus_pos_display)

        # Focus buttons
        focus_buttons = QHBoxLayout()
        self.focus_in_btn = QPushButton("Focus In")
        self.focus_in_btn.clicked.connect(lambda: self.adjust_focus(100))
        self.focus_out_btn = QPushButton("Focus Out")
        self.focus_out_btn.clicked.connect(lambda: self.adjust_focus(-100))
        focus_buttons.addWidget(self.focus_in_btn)
        focus_buttons.addWidget(self.focus_out_btn)
        control_layout.addLayout(focus_buttons)

        layout.addWidget(control_group)

        # Focus history graph
        self.focus_history_plot = pg.PlotWidget()
        self.setup_focus_history_plot()
        layout.addWidget(self.focus_history_plot)

        # Autofocus group
        autofocus_group = QGroupBox("Autofocus")
        autofocus_layout = QVBoxLayout(autofocus_group)

        self.autofocus_btn = QPushButton("Start Autofocus")
        self.autofocus_btn.clicked.connect(self.start_autofocus)
        autofocus_layout.addWidget(self.autofocus_btn)

        layout.addWidget(autofocus_group)

        return tab

    def create_rotate_position_tab(self):
        """Create the Rotate Position tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Rotator control group
        control_group = QGroupBox("Rotator Control")
        control_layout = QVBoxLayout(control_group)

        # Rotator position slider
        self.rotator_slider = QSlider(Qt.Horizontal)
        self.rotator_slider.setRange(0, 360)
        self.rotator_slider.setValue(0)
        self.rotator_slider.valueChanged.connect(self.set_rotator_position)
        control_layout.addWidget(self.rotator_slider)

        # Rotator position display
        self.rotator_pos_display = QLabel("0")
        self.rotator_pos_display.setAlignment(Qt.AlignCenter)
        self.rotator_pos_display.setStyleSheet("font-size: 24px; font-weight: bold;")
        control_layout.addWidget(self.rotator_pos_display)

        # Rotator buttons
        rotator_buttons = QHBoxLayout()
        self.rotator_cw_btn = QPushButton("Rotate CW")
        self.rotator_cw_btn.clicked.connect(lambda: self.adjust_rotator(-10))
        self.rotator_ccw_btn = QPushButton("Rotate CCW")
        self.rotator_ccw_btn.clicked.connect(lambda: self.adjust_rotator(10))
        rotator_buttons.addWidget(self.rotator_cw_btn)
        rotator_buttons.addWidget(self.rotator_ccw_btn)
        control_layout.addLayout(rotator_buttons)

        layout.addWidget(control_group)

        # Rotator history graph
        self.rotator_history_plot = pg.PlotWidget()
        self.setup_rotator_history_plot()
        layout.addWidget(self.rotator_history_plot)

        # Derotate group
        derotate_group = QGroupBox("Field Derotation")
        derotate_layout = QVBoxLayout(derotate_group)

        self.derotate_btn = QPushButton("Enable Derotation")
        self.derotate_btn.setCheckable(True)
        self.derotate_btn.clicked.connect(self.toggle_derotation)
        derotate_layout.addWidget(self.derotate_btn)

        layout.addWidget(derotate_group)

        return tab

    def create_mount_debug_tab(self):
        """Create the Mount Debug tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Debug controls group
        controls_group = QGroupBox("Debug Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.mount_debug_btn = QPushButton("Start Debug")
        self.mount_debug_btn.clicked.connect(self.start_mount_debug)
        controls_layout.addWidget(self.mount_debug_btn)

        self.mount_reset_btn = QPushButton("Reset Mount")
        self.mount_reset_btn.clicked.connect(self.reset_mount)
        controls_layout.addWidget(self.mount_reset_btn)

        layout.addWidget(controls_group)

        # Debug log
        debug_log_group = QGroupBox("Debug Log")
        debug_log_layout = QVBoxLayout(debug_log_group)

        self.mount_debug_log = QTextEdit()
        self.mount_debug_log.setReadOnly(True)
        debug_log_layout.addWidget(self.mount_debug_log)

        layout.addWidget(debug_log_group)

        # Debug plots
        debug_plots_group = QGroupBox("Debug Plots")
        debug_plots_layout = QVBoxLayout(debug_plots_group)

        self.mount_debug_plot = pg.PlotWidget()
        self.setup_mount_debug_plot()
        debug_plots_layout.addWidget(self.mount_debug_plot)

        layout.addWidget(debug_plots_group)

        return tab

    def create_visualization_panel(self):
        """Create the center visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Tab widget for different views
        tab_widget = QTabWidget()

        # Sky Map Tab
        sky_tab = QWidget()
        sky_layout = QVBoxLayout(sky_tab)

        # Create sky map widget
        self.sky_map = SkyMapTab()
        sky_layout.addWidget(self.sky_map)

        tab_widget.addTab(sky_tab, "Sky Map")
        tab_widget.addTab(self.create_target_tab(), "Target Coordinates")
        tab_widget.addTab(SearchTab(self.sky_map), "Star Search")

        # Camera Tab
        camera_tab = QWidget()
        camera_layout = QVBoxLayout(camera_tab)

        # Camera controls
        camera_controls = QHBoxLayout()

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 3600)
        self.exposure_spin.setValue(1.0)
        self.exposure_spin.setSuffix(" s")

        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_image)

        self.live_view_btn = QPushButton("Live View")
        self.live_view_btn.setCheckable(True)
        self.live_view_btn.clicked.connect(self.toggle_live_view)

        camera_controls.addWidget(QLabel("Exposure:"))
        camera_controls.addWidget(self.exposure_spin)
        camera_controls.addWidget(self.capture_btn)
        camera_controls.addWidget(self.live_view_btn)
        camera_controls.addStretch()

        camera_layout.addLayout(camera_controls)

        # Image display
        self.image_view = pg.ImageView()
        self.setup_image_view()
        camera_layout.addWidget(self.image_view)

        tab_widget.addTab(camera_tab, "Camera")

        layout.addWidget(tab_widget)
        return panel

    def create_status_panel(self):
        """Create the right status panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # System status group
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)

        # Time display
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #58a6ff;
            padding: 8px;
            background: rgba(88, 166, 255, 0.1);
            border-radius: 8px;
        """)
        status_layout.addWidget(self.time_label)

        # Position display
        pos_frame = QFrame()
        pos_frame.setFrameStyle(QFrame.StyledPanel)
        pos_layout = QFormLayout(pos_frame)

        self.current_ra = QLabel("12:00:00")
        self.current_dec = QLabel("+45:00:00")
        self.current_alt = QLabel("45.0")
        self.current_az = QLabel("180.0")
        pos_layout.addRow("Current RA:", self.current_ra)
        pos_layout.addRow("Current DEC:", self.current_dec)
        pos_layout.addRow("Altitude:", self.current_alt)
        pos_layout.addRow("Azimuth:", self.current_az)

        status_layout.addWidget(pos_frame)

        # Environmental data
        env_frame = QFrame()
        env_frame.setFrameStyle(QFrame.StyledPanel)
        env_layout = QFormLayout(env_frame)

        self.temperature = QLabel("15.0°C")
        self.humidity = QLabel("45%")
        self.pressure = QLabel("1013 hPa")
        self.wind_speed = QLabel("5.2 km/h")
        self.moon_phase = QLabel("Waxing Crescent (32%)")
        self.sidereal_time = QLabel("Sidereal Time: 00:00:00")
        env_layout.addRow("Sidereal Time:", self.sidereal_time)
        env_layout.addRow("Temperature:", self.temperature)
        env_layout.addRow("Humidity:", self.humidity)
        env_layout.addRow("Pressure:", self.pressure)
        env_layout.addRow("Wind Speed:", self.wind_speed)
        env_layout.addRow("Moon Phase:", self.moon_phase)

        status_layout.addWidget(env_frame)

        layout.addWidget(status_group)

        # Target list group
        targets_group = QGroupBox("Target List")
        targets_layout = QVBoxLayout(targets_group)

        self.target_list = QListWidget()
        self.setup_target_list()
        targets_layout.addWidget(self.target_list)

        # Target buttons
        target_buttons = QHBoxLayout()
        goto_target_btn = QPushButton("Go To")
        goto_target_btn.clicked.connect(self.goto_target)
        add_target_btn = QPushButton("Add")
        add_target_btn.clicked.connect(self.add_target)
        remove_target_btn = QPushButton("Remove")
        remove_target_btn.clicked.connect(self.remove_target)

        target_buttons.addWidget(goto_target_btn)
        target_buttons.addWidget(add_target_btn)
        target_buttons.addWidget(remove_target_btn)

        targets_layout.addLayout(target_buttons)

        layout.addWidget(targets_group)

        # Log group
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        # Log controls
        log_controls = QHBoxLayout()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.clear_log)
        save_log_btn = QPushButton("Save Log")
        save_log_btn.clicked.connect(self.save_log)

        log_controls.addWidget(clear_log_btn)
        log_controls.addWidget(save_log_btn)
        log_layout.addLayout(log_controls)

        layout.addWidget(log_group)

        return panel

    def setup_following_error_plot(self):
        """Configure the following error plot"""
        self.following_error_plot.setBackground('#0d1117')
        self.following_error_plot.setLabel('left', 'Error (arcsec)')
        self.following_error_plot.setLabel('bottom', 'Time (s)')
        self.following_error_plot.setTitle('Following Error')
        self.following_error_plot.showGrid(x=True, y=True, alpha=0.3)

        # Initialize with empty data
        self.ra_error_curve = self.following_error_plot.plot(
            pen="#f85149", name='RA Error'
        )
        self.dec_error_curve = self.following_error_plot.plot(
            pen="#58a6ff", name='DEC Error'
        )
        # Add legend
        self.following_error_plot.addLegend()

    def setup_focus_history_plot(self):
        """Configure the focus position history plot"""
        self.focus_history_plot.setBackground('#0d1117')
        self.focus_history_plot.setLabel('left', 'Focus Position')
        self.focus_history_plot.setLabel('bottom', 'Time (min)')
        self.focus_history_plot.setTitle('Focus Position History')
        self.focus_history_plot.showGrid(x=True, y=True, alpha=0.3)

        # Initialize with empty data
        self.focus_history_curve = self.focus_history_plot.plot(
            pen="#58a6ff", name="Focus Position"
        )

    def setup_rotator_history_plot(self):
        """Configure the rotator position history plot"""
        self.rotator_history_plot.setBackground('#0d1117')
        self.rotator_history_plot.setLabel('left', 'Rotator Position (°)')
        self.rotator_history_plot.setLabel('bottom', 'Time (min)')
        self.rotator_history_plot.setTitle('Rotator Position History')
        self.rotator_history_plot.showGrid(x=True, y=True, alpha=0.3)

        # Initialize with empty data
        self.rotator_history_curve = self.rotator_history_plot.plot(
            pen="#58a6ff", name="Rotator Position"
        )

    def setup_mount_debug_plot(self):
        """Configure the mount debug plot"""
        self.mount_debug_plot.setBackground('#0d1117')
        self.mount_debug_plot.setLabel('left', 'Value')
        self.mount_debug_plot.setLabel('bottom', 'Time (s)')
        self.mount_debug_plot.setTitle('Mount Debug Data')
        self.mount_debug_plot.showGrid(x=True, y=True, alpha=0.3)

        # Initialize with empty data
        self.mount_debug_curve1 = self.mount_debug_plot.plot(
            pen="#f85149", name="Motor Current"
        )
        self.mount_debug_curve2 = self.mount_debug_plot.plot(
            pen="#58a6ff", name="Temperature"
        )

        # Add legend
        self.mount_debug_plot.addLegend()

    def setup_image_view(self):
        """Configure the image view"""
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.getView().setBackgroundColor('#0d1117')

        # Set initial empty image
        empty_image = np.zeros((512, 512))
        self.image_view.setImage(empty_image)

    def setup_target_list(self):
        """Initialize the target list with sample data"""
        # Add sample targets
        targets = [
            "★★★ Vega (α Lyr)",
            "★★★ Altair (α Aql)",
            "★★★ Deneb (α Cyg)",
            "★★★ M31 (Andromeda Galaxy)",
            "★★★ M42 (Orion Nebula)",
            "★★★ M45 (Pleiades)",
            "★★★ Jupiter",
            "★★★ Saturn",
            "★★★ Mars",
            "★★★ Moon"
        ]

        for target in targets:
            self.target_list.addItem(target)

    def populate_catalog(self):
        """Populate the catalog table with sample data"""
        catalog_data = [
            ["Vega", "18h 36m 56.3s", "+38° 47' 01\"", "Star", 0.03],
            ["Altair", "19h 50m 47.8s", "+08° 52' 06\"", "Star", 0.77],
            ["Deneb", "20h 41m 25.9s", "+45° 16' 49\"", "Star", 1.25],
            ["M31", "08h 42m 44.3s", "+41° 16' 09\"", "Galaxy", 3.44],
            ["M42", "05h 35m 17.3s", "-05° 23' 28\"", "Nebula", 4.0],
            ["M45", "03h 47m 00.0s", "+24° 07' 00\"", "Cluster", 1.6],
            ["Jupiter", "22h 30m 00.0s", "-10° 00' 00\"", "Planet", -2.2],
            ["Saturn", "21h 00m 00.0s", "-17° 00' 00\"", "Planet", 0.5],
            ["Mars", "04h 00m 00.0s", "+20° 00' 00\"", "Planet", -1.0],
            ["Moon", "12h 00m 00.0s", "+00° 00' 00\"", "Moon", -12.7]
        ]

        self.catalog_table.setRowCount(len(catalog_data))
        for row, data in enumerate(catalog_data):
            for col, value in enumerate(data):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.catalog_table.setItem(row, col, item)

    def setup_timers(self):
        """Initialize system timers"""
        # Update timer for UI elements
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(1000)  # Update every second

        # Following error simulation timer
        self.following_error_timer = QTimer()
        self.following_error_timer.timeout.connect(self.update_following_error)

        # Live view simulation timer
        self.live_view_timer = QTimer()
        self.live_view_timer.timeout.connect(self.update_live_view)

        # Position history timer
        self.position_history_timer = QTimer()
        self.position_history_timer.timeout.connect(self.update_position_history)
        self.position_history_timer.start(60000)  # Update every minute

        # Sidereal time timer
        self.sidereal_timer = QTimer()
        self.sidereal_timer.timeout.connect(self.update_sidereal_time)
        self.sidereal_timer.start(1000)  # Update every second

    def update_ui(self):
        """Update all dynamic UI elements"""
        self.update_system_time()
        self.update_position_display()
        self.update_environmental_data()
        self.update_sky_map()

    def update_sidereal_time(self):
        """Update the sidereal time display"""
        now = Time.now()
        lst = now.sidereal_time('mean', longitude=self.backend.observatory_location.lon)
        self.sidereal_time.setText(f"Sidereal Time: {lst.to_string(sep=':')[0:11]}")

    def update_system_time(self):
        """Update the system time display"""
        current_time = Time.now().to_datetime(timezone=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.time_label.setText(current_time)
        self.system_time.setText(current_time)

    def update_position_display(self):
        """Update the position displays using backend data"""
        pos = self.backend.get_current_position()

        # Convert RA from degrees to HMS
        ra_h = int(pos.ra_deg / 15)
        ra_m = int((pos.ra_deg / 15 - ra_h) * 60)
        ra_s = int(((pos.ra_deg / 15 - ra_h) * 60 - ra_m) * 60)

        # Convert DEC from degrees to DMS
        dec_d = int(abs(pos.dec_deg))
        dec_m = int((abs(pos.dec_deg) - dec_d) * 60)
        dec_s = int(((abs(pos.dec_deg) - dec_d) * 60 - dec_m) * 60)
        dec_sign = "+" if pos.dec_deg >= 0 else "-"

        # Update displays
        ra_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:02d}"
        dec_str = f"{dec_sign}{dec_d:02d}:{dec_m:02d}:{dec_s:02d}"
        self.current_ra.setText(ra_str)
        self.current_dec.setText(dec_str)
        self.current_ra_display.setText(ra_str)
        self.current_dec_display.setText(dec_str)
        self.current_alt.setText(f"{pos.alt_deg:.1f}")
        self.current_az.setText(f"{pos.az_deg:.1f}")
        self.current_alt_display.setText(f"{pos.alt_deg:.1f}")
        self.current_az_display.setText(f"{pos.az_deg:.1f}")

        # Update target position displays
        target_pos = self.backend.target_position

        t_ra_h = int(target_pos.ra_deg / 15)
        t_ra_m = int((target_pos.ra_deg / 15 - t_ra_h) * 60)
        t_ra_s = int(((target_pos.ra_deg / 15 - t_ra_h) * 60 - t_ra_m) * 60)

        t_dec_d = int(abs(target_pos.dec_deg))
        t_dec_m = int((abs(target_pos.dec_deg) - t_dec_d) * 60)
        t_dec_s = int(((abs(target_pos.dec_deg) - t_dec_d) * 60 - t_dec_m) * 60)
        t_dec_sign = "+" if target_pos.dec_deg >= 0 else "-"

        t_ra_str = f"{t_ra_h:02d}:{t_ra_m:02d}:{t_ra_s:02d}"
        t_dec_str = f"{t_dec_sign}{t_dec_d:02d}:{t_dec_m:02d}:{t_dec_s:02d}"
        self.target_ra_display.setText(t_ra_str)
        self.target_dec_display.setText(t_dec_str)
        self.target_alt_display.setText(f"{target_pos.alt_deg:.1f}")
        self.target_az_display.setText(f"{target_pos.az_deg:.1f}")

    def update_environmental_data(self):
        """Update environmental sensors with simulated data"""
        env_data = self.backend.get_environmental_data()

        # Update displays
        self.temperature.setText(f"{env_data.temperature:.1f} °C")
        self.humidity.setText(f"{env_data.humidity:.0f} %")
        self.pressure.setText(f"{env_data.pressure:.1f} hPa")
        self.wind_speed.setText(f"{env_data.wind_speed:.1f} km/h")

        # Moon phase calculation (simplified)
        moon_phase = (time.time() / (29.53 * 86400)) % 1.0  # 29.53 day cycle
        if moon_phase < 0.25:
            phase_name = "Waxing Crescent"
            phase_percent = moon_phase / 0.25 * 100
        elif moon_phase < 0.5:
            phase_name = "Waxing Gibbous"
            phase_percent = (moon_phase - 0.25) / 0.25 * 100 + 25
        elif moon_phase < 0.75:
            phase_name = "Waning Gibbous"
            phase_percent = (moon_phase - 0.5) / 0.25 * 100 + 50
        else:
            phase_name = "Waning Crescent"
            phase_percent = (moon_phase - 0.75) / 0.25 * 100 + 75

        self.moon_phase.setText(f"{phase_name} ({phase_percent:.0f}%)")

    def update_sky_map(self):
        """Update the sky map with current and target positions"""
        pos = self.backend.get_current_position()
        target_pos = self.backend.target_position

        self.sky_map.set_current_position(pos.ra_deg, pos.dec_deg)
        self.sky_map.set_target_position(target_pos.ra_deg, target_pos.dec_deg)

    def update_following_error(self):
        """Update the following error plot with data from backend"""
        if self.backend.is_tracking:
            # For OPC UA, we don't have tracking error data, so we'll simulate it
            time_data = np.arange(0, 10, 0.1)
            ra_error = np.sin(time_data) * 0.5 + np.random.normal(0, 0.1, len(time_data))
            dec_error = np.cos(time_data) * 0.3 + np.random.normal(0, 0.05, len(time_data))

            self.ra_error_curve.setData(time_data, ra_error)
            self.dec_error_curve.setData(time_data, dec_error)

    def update_position_history(self):
        """Update the mount position history"""
        pos = self.backend.get_current_position()
        timestamp = time.time() / 60  # Minutes since epoch

        # In a real system, we would store position history
        # For now, we'll just update the display with current position
        self.ra_history_curve.setData([timestamp], [pos.ra_deg])
        self.dec_history_curve.setData([timestamp], [pos.dec_deg])

    def update_live_view(self):
        """Update the live view with simulated frames"""
        if self.live_view_active:
            # Generate random noise with occasional stars
            image_data = np.random.poisson(10, (512, 512)) + np.random.normal(0, 5, (512, 512))

            # Add some random stars
            for _ in range(5):
                x, y = np.random.randint(0, 512, 2)
                brightness = np.random.randint(100, 500)
                image_data[max(0, y-1):min(512, y+2), max(0, x-1):min(512, x+2)] += brightness

            self.image_view.setImage(image_data)

    def update_connection_indicator(self):
        """Update the connection status indicator"""
        if self.connected:
            self.connection_indicator.setText(" ▲ Connected")
            self.connection_indicator.setStyleSheet("color: #238636; font-weight: bold;")
        else:
            self.connection_indicator.setText(" ▼ Disconnected")
            self.connection_indicator.setStyleSheet("color: #f85149; font-weight: bold;")

    def log_message(self, message):
        """Add a message to the system log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f'[{timestamp}] {message}')
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())
        # Also log to console for debugging
        logger.info(message)

    def clear_log(self):
        """Clear the system log"""
        self.log_text.clear()
        self.log_message("Log cleared")

    def save_log(self):
        """Save the log to a file"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Log File", "", "Text Files (*.txt);All Files (*)", options=options
        )
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"Log saved to {file_name}")
            except Exception as e:
                self.log_message(f"Error saving log: {str(e)}")

    def toggle_connection(self):
        """Toggle telescope connection state"""
        if not self.connected:
            # Connect to telescope
            if self.backend.connect():
                self.connected = True
                self.connect_btn.setText("Disconnect")
                self.connection_status.setText("Status: Connected")
                self.update_connection_indicator()
                self.log_message("Telescope connected successfully")

                # Enable controls
                self.enable_controls(True)
        else:
            # Disconnect from telescope
            if self.backend.disconnect():
                self.connected = False
                self.connect_btn.setText("Connect Telescope")
                self.connection_status.setText("Status: Disconnected")
                self.update_connection_indicator()
                self.log_message("Telescope disconnected")

                # Disable controls
                self.enable_controls(False)

    def enable_controls(self, enabled):
        """Enable or disable telescope controls based on connection state"""
        self.tracking_btn.setEnabled(enabled)
        self.tracking_rate_combo.setEnabled(enabled and not self.backend.is_tracking)
        self.guide_btn.setEnabled(enabled)
        self.park_btn.setEnabled(enabled)
        self.home_btn.setEnabled(enabled)
        self.unpark_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.focus_slider.setEnabled(enabled)
        self.focus_in_btn.setEnabled(enabled)
        self.focus_out_btn.setEnabled(enabled)
        self.autofocus_btn.setEnabled(enabled)
        self.rotator_slider.setEnabled(enabled)
        self.rotator_cw_btn.setEnabled(enabled)
        self.rotator_ccw_btn.setEnabled(enabled)
        self.derotate_btn.setEnabled(enabled)
        self.capture_btn.setEnabled(enabled)
        self.live_view_btn.setEnabled(enabled)
        self.goto_catalog_btn.setEnabled(enabled)
        self.add_to_list_btn.setEnabled(enabled)
        self.goto_target_btn.setEnabled(enabled)
        self.add_target_btn.setEnabled(enabled)
        self.remove_target_btn.setEnabled(enabled)
        self.goto_coord_btn.setEnabled(enabled)
        self.ra_input.setEnabled(enabled)
        self.dec_input.setEnabled(enabled)
        self.target_ra_input.setEnabled(enabled)
        self.target_dec_input.setEnabled(enabled)
        self.get_altaz_btn.setEnabled(enabled)
        self.goto_altaz_btn.setEnabled(enabled)
        self.save_altaz_btn.setEnabled(enabled)

    def toggle_tracking(self):
        """Toggle telescope tracking state"""
        if self.backend.is_tracking:
            # Stop tracking
            if self.backend.stop_tracking():
                self.tracking_btn.setText("Start Tracking")
                self.tracking = False
                self.log_message("Tracking stopped")
        else:
            # Start tracking
            rate = TrackingRate[self.tracking_rate_combo.currentText().upper()]
            if self.backend.start_tracking(rate):
                self.tracking_btn.setText("Stop Tracking")
                self.tracking = True
                self.log_message(f"Tracking started ({self.tracking_rate_combo.currentText()} rate)")

    def toggle_guiding(self):
        """Toggle auto-guiding state"""
        if not self.guiding:
            # Start guiding
            self.guiding = True
            self.guide_btn.setText("Stop Guiding")
            self.guide_status.setText("Guide Status: Active")
            self.backend.is_guiding = True
            self.log_message("Auto-guiding started")
        else:
            # Stop guiding
            self.guiding = False
            self.guide_btn.setText("Start Guiding")
            self.guide_status.setText("Guide Status: Stopped")
            self.backend.is_guiding = False
            self.log_message("Auto-guiding stopped")

    def update_tracking_rate(self, rate):
        """Update the tracking rate"""
        self.tracking_rate = rate
        if self.tracking:
            self.backend.tracking_rate = TrackingRate[rate.upper()]
            self.log_message(f"Tracking rate changed to {rate}")

    def park_telescope(self):
        """Park the telescope"""
        if self.connected:
            if self.backend.park():
                self.log_message("Telescope parking initiated")

    def home_telescope(self):
        """Home the telescope"""
        if self.connected:
            if self.backend.home():
                self.log_message("Telescope homing initiated")

    def unpark_telescope(self):
        """Unpark the telescope"""
        if self.connected:
            self.log_message("Telescope unparked")
            self.backend.telescope_state = TelescopeState.IDLE

    def emergency_stop(self):
        """Emergency stop all telescope motion"""
        if self.connected:
            if self.backend.stop_motion():
                self.log_message("Emergency stop activated")
                self.tracking = False
                self.tracking_btn.setText("Start Tracking")
                self.following_error_timer.stop()

    def search_catalog(self):
        """Search the catalog for objects"""
        search_term = self.search_input.text().strip()
        if not search_term:
            return

        results = self.backend.search_catalog(search_term)
        self.catalog_table.setRowCount(len(results))

        for row, obj in enumerate(results):
            # Convert RA/DEC to display format
            ra_h = int(obj.ra_deg / 15)
            ra_m = int((obj.ra_deg / 15 - ra_h) * 60)
            ra_s = int(((obj.ra_deg / 15 - ra_h) * 60 - ra_m) * 60)

            # Convert DEC from degrees to DMS
            dec_d = int(abs(obj.dec_deg))
            dec_m = int((abs(obj.dec_deg) - dec_d) * 60)
            dec_s = int(((abs(obj.dec_deg) - dec_d) * 60 - dec_m) * 60)
            dec_sign = "+" if obj.dec_deg >= 0 else "-"

            ra_str = f"{ra_h:02d}h {ra_m:02d}m {ra_s:02d}s"
            dec_str = f"{dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:02d}\""

            # Add items to table
            self.catalog_table.setItem(row, 0, QTableWidgetItem(obj.name))
            self.catalog_table.setItem(row, 1, QTableWidgetItem(ra_str))
            self.catalog_table.setItem(row, 2, QTableWidgetItem(dec_str))
            self.catalog_table.setItem(row, 3, QTableWidgetItem(obj.obj_type))
            self.catalog_table.setItem(row, 4, QTableWidgetItem(str(obj.magnitude)))

    def goto_catalog_object(self):
        """Slew to the currently selected catalog object"""
        selected = self.catalog_table.currentRow()
        if selected >= 0:
            obj = self.backend.catalog[selected]
            if self.backend.slew_to(obj.ra_deg, obj.dec_deg):
                self.log_message(f"Slewing to {obj.name} at RA: {obj.ra_deg:.2f}, DEC: {obj.dec_deg:.2f}")

    def add_catalog_to_list(self):
        """Add the currently selected catalog object to target list"""
        selected = self.catalog_table.currentRow()
        if selected >= 0:
            obj = self.backend.catalog[selected]
            self.target_list.addItem(f"★ {obj.name}")
            self.log_message(f"Added {obj.name} to target list")

    def goto_target(self):
        """Slew to the currently selected target"""
        selected = self.target_list.currentRow()
        if selected >= 0:
            target_name = self.target_list.currentItem().text()
            # Find matching object in catalog
            for obj in self.backend.catalog:
                if obj.name in target_name:
                    if self.backend.slew_to(obj.ra_deg, obj.dec_deg):
                        self.log_message(f"Slewing to {obj.name} at RA: {obj.ra_deg:.2f}, DEC: {obj.dec_deg:.2f}")
                    break

    def add_target(self):
        """Add a new target to the list"""
        text, ok = QInputDialog.getText(self, "Add Target", "Enter target name:")
        if ok and text:
            self.target_list.addItem(text)
            self.log_message(f"Added {text} to target list")

    def remove_target(self):
        """Remove the selected target from the list"""
        selected = self.target_list.currentRow()
        if selected >= 0:
            item = self.target_list.takeItem(selected)
            self.log_message(f"Removed {item.text()} from target list")

    def parse_coordinate(self, coord_str: str, is_ra: bool = False) -> float:
        """Parse a coordinate string (RA or DEC) into degrees"""
        coord_str = coord_str.strip()
        self.log_message(f"Parsing coordinate: {coord_str} (is_ra={is_ra})")

        # Try to parse as decimal degrees first
        try:
            val = float(coord_str)
            self.log_message(f"Interpreted as decimal degrees: {val}")
            return val
        except ValueError:
            pass

        # Try to parse as HMS or DMS
        parts = []
        for sep in [':', 'h', 'm', 's', '°', "'", '"', 'd']:
            if sep in coord_str:
                parts = coord_str.split(sep)
                break

        if not parts:
            # No separators found, try splitting by space
            parts = coord_str.split()

        if len(parts) == 1:
            # Single value, assume degrees
            try:
                val = float(parts[0])
                self.log_message(f"Interpreted as single value degrees: {val}")
                return val
            except ValueError:
                self.log_message("Could not parse coordinate")
                return 0.0

        # Parse hours/minutes/seconds or degrees/arcminutes/arcseconds
        try:
            if is_ra:
                # RA format (HH:MM:SS or HH MM SS)
                h = float(parts[0])
                m = float(parts[1]) if len(parts) > 1 else 0.0
                s = float(parts[2]) if len(parts) > 2 else 0.0
                val = (h + m/60 + s/3600) * 15  # Convert to degrees
                self.log_message(f"Interpreted as HMS: {h}h {m}m {s}s = {val}°")
                return val
            else:
                # DEC format (DD:MM:SS or DD MM SS)
                d = float(parts[0])
                m = float(parts[1]) if len(parts) > 1 else 0.0
                s = float(parts[2]) if len(parts) > 2 else 0.0
                sign = 1 if d >= 0 else -1
                val = sign * (abs(d) + m/60 + s/3600)
                self.log_message(f"Interpreted as DMS: {d}° {m}' {s}\" = {val}°")
                return val
        except ValueError:
            self.log_message("Could not parse coordinate components")
            return 0.0

    def goto_coordinates(self):
        """Slew to manually entered coordinates"""
        if not self.connected:
            return

        ra_str = self.ra_input.text()
        dec_str = self.dec_input.text()

        if not ra_str or not dec_str:
            self.log_message("Please enter both RA and DEC coordinates")
            return

        ra_deg = self.parse_coordinate(ra_str, is_ra=True)
        dec_deg = self.parse_coordinate(dec_str, is_ra=False)

        if ra_deg < 0 or ra_deg > 360:
            self.log_message("RA must be between 0 and 360 degrees")
            return

        if dec_deg < -90 or dec_deg > 90:
            self.log_message("DEC must be between -90 and +90 degrees")
            return

        if self.backend.slew_to(ra_deg, dec_deg):
            self.log_message(f"Slewing to RA: {ra_deg:.2f}, DEC: {dec_deg:.2f}")

    def get_altaz(self):
        """Calculate Alt/Az from RA/DEC using Astropy"""
        ra_str = self.target_ra_input.text()
        dec_str = self.target_dec_input.text()

        if not ra_str or not dec_str:
            self.log_message("Please enter both RA and DEC coordinates")
            return

        ra_deg = self.parse_coordinate(ra_str, is_ra=True)
        dec_deg = self.parse_coordinate(dec_str, is_ra=False)

        if ra_deg < 0 or ra_deg > 360:
            self.log_message("RA must be between 0 and 360 degrees")
            return

        if dec_deg < -90 or dec_deg > 90:
            self.log_message("DEC must be between -90 and +90 degrees")
            return

        # Create SkyCoord object
        target_coord = SkyCoord(
            ra=ra_deg * u.deg,
            dec=dec_deg * u.deg,
            frame='icrs'
        )

        # Get Alt/Az
        alt, az = self.backend.get_altaz(target_coord)

        self.target_alt_display.setText(f"{alt:.4f}")
        self.target_az_display.setText(f"{az:.4f}")
        self.log_message(f"Calculated Alt/Az: {alt:.2f}, {az:.2f}")
        self.log_message(f"Using observatory location: Lat={self.backend.observatory_location.lat}, Lon={self.backend.observatory_location.lon}")

    def goto_altaz(self):
        """Slew to manually entered Alt/Az"""
        try:
            alt_deg = float(self.target_alt_display.text())
            az_deg = float(self.target_az_display.text())
        except ValueError:
            self.log_message("Invalid Alt/Az values")
            return

        if alt_deg < 0 or alt_deg > 90:
            self.log_message("Altitude must be between 0 and 90 degrees")
            return

        if az_deg < 0 or az_deg > 360:
            self.log_message("Azimuth must be between 0 and 360 degrees")
            return

        # Create AltAz coordinates
        altaz = AltAz(
            alt=alt_deg * u.deg,
            az=az_deg * u.deg,
            obstime=Time.now(),
            location=self.backend.observatory_location
        )

        # Convert to ICRS (RA/DEC)
        skycoord = SkyCoord(altaz.transform_to(ICRS()))

        self.log_message(f"Converted Alt/Az {alt_deg:.2f}, {az_deg:.2f} to RA/DEC: {skycoord.ra.degree:.2f}, {skycoord.dec.degree:.2f}")

        if self.connected:
            if self.backend.slew_to(skycoord.ra.degree, skycoord.dec.degree):
                self.log_message(f"Slewing to Alt/Az: {alt_deg:.2f}, {az_deg:.2f} (RA: {skycoord.ra.degree:.2f}, DEC: {skycoord.dec.degree:.2f})")
        else:
            self.log_message("Not connected - computation only (no telescope movement)")
    def save_altaz(self):
        """Save the current Alt/Az coordinates"""
        try:
            alt = float(self.target_alt_display.text())
            az = float(self.target_az_display.text())
            self.log_message(f"Saved Alt/Az: {alt:.2f}, {az:.2f}")
        except ValueError:
            self.log_message("No valid Alt/Az coordinates to save")

    def set_focus_position(self, position):
        """Set the focus position"""
        self.focus_pos_display.setText(str(position))
        # In a real system, this would send the command to the focuser
        self.log_message(f"Focus position set to {position}")

    def adjust_focus(self, delta):
        """Adjust focus by specified amount"""
        current_pos = int(self.focus_pos_display.text())
        new_pos = max(0, min(10000, current_pos + delta))
        self.focus_slider.setValue(new_pos)
        self.log_message(f"Focus adjusted by {delta} to {new_pos}")

    def start_autofocus(self):
        """Start autofocus routine"""
        self.log_message("Starting autofocus routine...")
        # Simulate autofocus by moving through positions
        for pos in range(4000, 6001, 200):
            self.focus_slider.setValue(pos)
            time.sleep(0.1)
        self.focus_slider.setValue(5000)
        self.log_message("Autofocus completed")

    def set_rotator_position(self, position):
        """Set the rotator position"""
        position = position % 360  # Wrap around at 360 degrees
        self.rotator_pos_display.setText(str(position))
        # In a real system, this would send the command to the rotator
        self.log_message(f"Rotator position set to {position}°")

    def adjust_rotator(self, delta):
        """Adjust rotator by specified amount"""
        current_pos = int(self.rotator_pos_display.text())
        new_pos = (current_pos + delta) % 360
        self.rotator_slider.setValue(new_pos)
        self.log_message(f"Rotator adjusted by {delta}° to {new_pos}°")

    def toggle_derotation(self, checked):
        """Toggle field derotation"""
        if checked:
            self.log_message("Field derotation enabled")
            self.derotate_btn.setText("Disable Derotation")
        else:
            self.log_message("Field derotation disabled")
            self.derotate_btn.setText("Enable Derotation")

    def start_mount_debug(self):
        """Start mount debug routine"""
        self.log_message("Starting mount debug...")
        # Simulate debug data collection
        self.mount_debug_log.append("Collecting mount data...")
        self.mount_debug_log.append("Motor currents: 1.2A, 1.1A")
        self.mount_debug_log.append("Temperatures: 32°C, 34°C")
        self.mount_debug_log.append("Encoders: nominal")
        self.log_message("Mount debug completed")

    def reset_mount(self):
        """Reset mount controller"""
        self.log_message("Resetting mount controller...")
        # Simulate reset
        time.sleep(1)
        self.log_message("Mount reset complete")

    def capture_image(self):
        """Capture an image with current settings"""
        if not self.connected:
            return

        exposure = self.exposure_spin.value()
        self.log_message(f"Capturing {exposure:.1f}s exposure...")

        # Simulate image capture
        time.sleep(exposure)

        # Generate a simulated image
        image_data = np.random.poisson(10, (512, 512)) + np.random.normal(0, 5, (512, 512))

        # Add some stars
        for _ in range(20):
            x, y = np.random.randint(0, 512, 2)
            brightness = np.random.randint(100, 1000)
            image_data[max(0, y-2):min(512, y+3), max(0, x-2):min(512, x+3)] += brightness

        self.image_view.setImage(image_data)
        self.log_message(f"Capture complete ({exposure:.1f}s)")

    def toggle_live_view(self, checked):
        """Toggle live view mode"""
        if checked:
            self.live_view_active = True
            self.live_view_timer.start(100)  # 10 fps
            self.log_message("Live view started")
        else:
            self.live_view_active = False
            self.live_view_timer.stop()
            self.log_message("Live view stopped")

    def closeEvent(self, event):
        """Handle application close event"""
        # Clean up resources
        if self.connected:
            self.backend.disconnect()

        # Stop all timers
        self.update_timer.stop()
        self.following_error_timer.stop()
        self.live_view_timer.stop()
        self.position_history_timer.stop()
        self.sidereal_timer.stop()

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = TelescopeControlSystem()
    window.show()

    # Start application
    sys.exit(app.exec_())
